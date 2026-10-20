from __future__ import annotations

import json
import logging
import os
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("vllm_gateway")

ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MIN_CONTEXT_LENGTH = 1024
_MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
_MODEL_ARCHITECTURE_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
_QWEN3_HYBRID = re.compile(r"^qwen3_(?:\d+|next)")


@dataclass(frozen=True, slots=True)
class FamilyProfile:
    """Parsers that vLLM needs for a model family."""

    reasoning_parser: str | None = None
    tool_call_parser: str | None = None


def profile_for(model_types: Iterable[str]) -> FamilyProfile:
    for model_type in model_types:
        if _QWEN3_HYBRID.match(model_type):
            return FamilyProfile("qwen3", "qwen3_coder")
        if model_type in {"qwen3", "qwen3_moe"}:
            return FamilyProfile("qwen3", "hermes")
    return FamilyProfile()


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A complete, loadable model snapshot found in the Hugging Face cache."""

    repo_id: str
    snapshot: Path
    model_types: tuple[str, ...]
    max_context: int
    weights_bytes: int
    kv_bytes_per_token: int
    mtp: bool

    @property
    def profile(self) -> FamilyProfile:
        return profile_for(self.model_types)

    def as_dict(self) -> dict[str, object]:
        profile = self.profile
        return {
            "repo_id": self.repo_id,
            "size_bytes": self.weights_bytes,
            "max_context": self.max_context,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "mtp": self.mtp,
            "reasoning_parser": profile.reasoning_parser,
            "tool_call_parser": profile.tool_call_parser,
        }


def _safetensors_keys(path: Path) -> set[str]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > _MAX_SAFETENSORS_HEADER:
            raise ValueError(f"safetensors header is too large: {path}")
        header = json.loads(f.read(length))
    return {key for key in header if key != "__metadata__"}


def _weights(snapshot: Path) -> tuple[list[Path], set[str]] | None:
    """Return weight files and tensor names, or None when the snapshot is incomplete."""

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        paths = [snapshot / name for name in sorted(set(weight_map.values()))]
        if not paths or not all(path.exists() for path in paths):
            return None
        return paths, set(weight_map)
    paths = sorted(snapshot.glob("*.safetensors"))
    if not paths:
        return None
    keys: set[str] = set()
    for path in paths:
        keys |= _safetensors_keys(path)
    return paths, keys


def _positive_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def inspect_snapshot(repo_id: str, snapshot: Path) -> ModelInfo | None:
    """Read a snapshot's config and weights. Return None when it cannot be served."""

    if any((snapshot.parent.parent / "blobs").glob("*.incomplete")):
        return None
    config_file = snapshot / "config.json"
    if not config_file.is_file():
        return None
    config = json.loads(config_file.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if not any(str(name).endswith(_MODEL_ARCHITECTURE_SUFFIXES) for name in architectures):
        return None
    weights = _weights(snapshot)
    if weights is None:
        return None
    paths, keys = weights

    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    max_context = _positive_int(text.get("max_position_embeddings")) or _positive_int(
        config.get("max_position_embeddings")
    )
    layers = _positive_int(text.get("num_hidden_layers"))
    heads = _positive_int(text.get("num_attention_heads"))
    kv_heads = _positive_int(text.get("num_key_value_heads")) or heads
    hidden = _positive_int(text.get("hidden_size"))
    head_dim = _positive_int(text.get("head_dim")) or (hidden // heads if heads else 0)
    if not (max_context and layers and kv_heads and head_dim):
        return None

    layer_types = text.get("layer_types")
    interval = _positive_int(text.get("full_attention_interval"))
    if isinstance(layer_types, list):
        full_layers = sum(item == "full_attention" for item in layer_types)
    elif interval:
        full_layers = layers // interval
    else:
        full_layers = layers

    mtp_layers = _positive_int(text.get("mtp_num_hidden_layers")) or _positive_int(text.get("num_nextn_predict_layers"))
    mtp = bool(mtp_layers) and any(key.startswith("mtp.") or ".mtp." in key for key in keys)
    if mtp:
        full_layers += mtp_layers

    model_types = tuple(
        str(item) for item in (config.get("model_type"), text.get("model_type")) if isinstance(item, str) and item
    )
    return ModelInfo(
        repo_id=repo_id,
        snapshot=snapshot.resolve(),
        model_types=model_types,
        max_context=max_context,
        weights_bytes=sum(path.stat().st_size for path in paths),
        kv_bytes_per_token=2 * full_layers * kv_heads * head_dim,
        mtp=mtp,
    )


def _snapshot_dir(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    if not snapshots.is_dir():
        return None
    candidates = [item for item in snapshots.iterdir() if item.is_dir()]
    return max(candidates, key=lambda item: item.stat().st_mtime, default=None)


def _inspect_repo(repo_dir: Path) -> ModelInfo | None:
    snapshot = _snapshot_dir(repo_dir)
    if snapshot is None:
        return None
    repo_id = repo_dir.name.removeprefix("models--").replace("--", "/")
    return inspect_snapshot(repo_id, snapshot)


def _repo_dir(hub_cache: Path, repo_id: str) -> Path:
    return hub_cache / f"models--{repo_id.replace('/', '--')}"


def find_local_model(hub_cache: Path, repo_id: str) -> ModelInfo | None:
    try:
        return _inspect_repo(_repo_dir(hub_cache, repo_id))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        LOGGER.warning("cannot inspect %s in %s: %s", repo_id, hub_cache, exc)
        return None


def scan_local_models(hub_cache: Path) -> list[ModelInfo]:
    """List loadable models. Only `models--*` directories are visited and unreadable ones are skipped."""

    try:
        repo_dirs = sorted(hub_cache.glob("models--*"))
    except OSError:
        return []
    found: list[ModelInfo] = []
    for repo_dir in repo_dirs:
        try:
            info = _inspect_repo(repo_dir)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            LOGGER.debug("skipping %s: %s", repo_dir, exc)
            continue
        if info is not None:
            found.append(info)
    return found


def default_hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A user alias bound to one model snapshot and one context length."""

    name: str
    model_path: str
    repo_id: str = ""
    context_length: int | None = None
    weights_bytes: int = 0
    kv_bytes_per_token: int = 0
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    mtp: bool = False

    def public_dict(self, loaded: bool = False) -> dict[str, object]:
        return {
            "id": self.name,
            "object": "model",
            "owned_by": "local",
            "created": 0,
            "loaded": loaded,
            "metadata": {"repo_id": self.repo_id, "context_length": self.context_length},
        }


def _spec_from_info(alias: str, info: ModelInfo, context_length: int) -> ModelSpec:
    profile = info.profile
    return ModelSpec(
        name=alias,
        model_path=str(info.snapshot),
        repo_id=info.repo_id,
        context_length=context_length,
        weights_bytes=info.weights_bytes,
        kv_bytes_per_token=info.kv_bytes_per_token,
        reasoning_parser=profile.reasoning_parser,
        tool_call_parser=profile.tool_call_parser,
        mtp=info.mtp,
    )


class ModelCatalog:
    """User aliases, persisted to `state_file` and resolved against the Hugging Face cache."""

    def __init__(
        self,
        specs: Iterable[ModelSpec] = (),
        *,
        state_file: Path | None = None,
        hub_cache: Path | None = None,
    ):
        self.state_file = state_file
        self.hub_cache = hub_cache
        self._specs: dict[str, ModelSpec] = {}
        self._unresolved: dict[str, dict[str, Any]] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate alias {spec.name!r}")
            self._specs[spec.name] = spec

    @classmethod
    def load(cls, state_file: Path, hub_cache: Path) -> ModelCatalog:
        catalog = cls(state_file=state_file, hub_cache=hub_cache)
        if not state_file.is_file():
            return catalog
        try:
            entries = json.loads(state_file.read_text(encoding="utf-8"))["aliases"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid alias state file {state_file}: {exc}") from exc
        for alias, entry in entries.items():
            try:
                catalog._specs[alias] = catalog._resolve(alias, entry["repo_id"], entry["context_length"])
            except (KeyError, ValueError, TypeError) as exc:
                LOGGER.warning("alias %r is unavailable and kept unloaded: %s", alias, exc)
                catalog._unresolved[alias] = entry
        return catalog

    def _resolve(self, alias: str, repo_id: str, context_length: object) -> ModelSpec:
        if not isinstance(alias, str) or not ALIAS_PATTERN.fullmatch(alias):
            raise ValueError("alias must match [A-Za-z0-9._-]{1,64}")
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError("model must be a Hugging Face repo id")
        if self.hub_cache is None:
            raise RuntimeError("catalog has no Hugging Face cache")
        info = find_local_model(self.hub_cache, repo_id.strip())
        if info is None:
            raise KeyError(f"model {repo_id!r} is not available in the local cache")
        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise ValueError("context_length must be an integer")
        if not MIN_CONTEXT_LENGTH <= context_length <= info.max_context:
            raise ValueError(
                f"context_length must be between {MIN_CONTEXT_LENGTH} and {info.max_context} for {info.repo_id}"
            )
        return _spec_from_info(alias, info, context_length)

    def add(self, alias: str, repo_id: str, context_length: object) -> ModelSpec:
        if alias in self._specs or alias in self._unresolved:
            raise ValueError(f"alias {alias!r} already exists")
        spec = self._resolve(alias, repo_id, context_length)
        self._specs[alias] = spec
        self._save()
        return spec

    def remove(self, alias: str) -> ModelSpec:
        spec = self.get(alias)
        del self._specs[alias]
        self._save()
        return spec

    def _save(self) -> None:
        if self.state_file is None:
            return
        entries = {
            **self._unresolved,
            **{
                name: {"repo_id": spec.repo_id, "context_length": spec.context_length}
                for name, spec in self._specs.items()
            },
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(json.dumps({"aliases": entries}, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return tuple(self._specs.values())

    def get(self, name: str | None) -> ModelSpec:
        spec = self._specs.get(name.strip()) if isinstance(name, str) else None
        if spec is None:
            available = ", ".join(self._specs) or "none"
            raise KeyError(f"unknown model {name!r}; available models: {available}")
        return spec

    def public_models(self, loaded_names: str | Iterable[str] | None = None) -> list[dict[str, object]]:
        loaded = {loaded_names} if isinstance(loaded_names, str) else set(loaded_names or ())
        return [spec.public_dict(spec.name in loaded) for spec in self._specs.values()]
