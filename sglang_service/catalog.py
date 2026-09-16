from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        return int(s) if s else None
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        return float(s) if s else None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def _parse_kv_cache_dtype(
    cache_k: str | None = None,
    cache_v: str | None = None,
    explicit_dtype: str | None = None,
) -> str | None:
    if explicit_dtype and str(explicit_dtype).strip():
        dtype = str(explicit_dtype).strip().lower().replace("-", "_")
    else:
        dtype = (cache_k or cache_v or "").strip().lower().replace("-", "_")
    if not dtype:
        return None
    if dtype in {"q8_0", "q8", "fp8", "fp8_e5m2"}:
        return "fp8_e5m2"
    if dtype in {"fp8_e4m3"}:
        return "fp8_e4m3"
    if dtype in {"fp4", "fp4_e2m1", "q4_0"}:
        return "fp4_e2m1"
    if dtype in {"bf16", "bfloat16"}:
        return "bfloat16"
    if dtype in {"fp16", "f16", "auto"}:
        return "auto"
    return dtype


def _parse_preset_file(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Parse legacy llama.cpp INI dialect for backward compatibility."""

    if not path.is_file():
        raise FileNotFoundError(f"preset file does not exist: {path}")

    root: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] = root
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"empty section at {path}:{line_number}")
            current = sections.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"expected key=value at {path}:{line_number}")
        key, value = line.split("=", 1)
        current[key.strip().lower()] = value.strip()

    return root, sections


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Specification for a model exposed by the vLLM gateway."""

    name: str
    model_path: str = ""
    hf_repo: str | None = None
    hf_file: str | None = None
    context_length: int | None = None
    kv_cache_dtype: str | None = None
    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    device_ids: tuple[int, ...] | None = None
    async_scheduling: bool | None = None
    flashinfer_autotune: bool | None = None
    enable_prefix_caching: bool | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    embeddings: bool = False
    pooling: str | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    spec_type: str | None = None
    rope_scaling: str | None = None
    rope_scale: float | None = None
    yarn_orig_ctx: int | None = None
    rope_theta: float | None = None
    idle_seconds: float | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    chat_template: str | None = None
    attention_backend: str | None = None
    vllm_args: tuple[str, ...] = ()
    is_default: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def source(self) -> str:
        if self.model_path:
            return self.model_path
        if self.hf_repo and self.hf_file:
            return f"{self.hf_repo}/{self.hf_file}"
        return self.hf_repo or ""

    @property
    def model(self) -> str:
        return self.model_path or self.source

    @property
    def is_gguf(self) -> bool:
        src = (self.hf_file or self.model_path or self.hf_repo or "").lower()
        return src.endswith(".gguf")

    def public_dict(self, loaded: bool = False) -> dict[str, object]:
        return {
            "id": self.name,
            "object": "model",
            "owned_by": "local",
            "created": 0,
            "loaded": loaded,
            "metadata": {
                "source": self.source,
                "format": "gguf" if self.is_gguf else "safetensors",
                "embedding": self.embeddings,
                "pooling": self.pooling,
                "reasoning": self.reasoning,
                "reasoning_budget": self.reasoning_budget,
                "speculative_type": self.spec_type,
                "kv_cache_dtype": self.kv_cache_dtype,
                "tensor_parallel_size": self.tensor_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "rope_scaling": self.rope_scaling,
                "rope_scale": self.rope_scale,
                "yarn_orig_ctx": self.yarn_orig_ctx,
            },
        }


class ModelCatalog:
    def __init__(self, specs: Iterable[ModelSpec], config_file: Path):
        self._specs = tuple(specs)
        if not self._specs:
            raise ValueError(f"no named models found in configuration file: {config_file}")
        self.config_file = config_file
        self.preset_file = config_file  # backward compatibility alias
        self._by_name: dict[str, ModelSpec] = {}
        self._by_normalized: dict[str, ModelSpec] = {}
        self._by_alias: dict[str, ModelSpec] = {}
        lookup_owners: dict[str, tuple[ModelSpec, str]] = {}

        def register(value: str, spec: ModelSpec, kind: str, *, canonical: bool = False) -> None:
            raw = value.strip()
            if not raw:
                return
            normalized = self._normalize(raw)
            existing = lookup_owners.get(normalized)
            if existing and existing[0] is not spec:
                other, other_kind = existing
                raise ValueError(
                    f"model lookup collision for {raw!r}: {spec.name!r} ({kind}) "
                    f"conflicts with {other.name!r} ({other_kind})"
                )
            lookup_owners[normalized] = (spec, kind)
            if canonical:
                self._by_name[raw] = spec
                self._by_normalized[normalized] = spec
            else:
                self._by_alias[raw] = spec
                self._by_alias[raw.lower()] = spec
                self._by_alias[normalized] = spec

        for spec in self._specs:
            register(spec.name, spec, "canonical name", canonical=True)
            for alias in spec.aliases:
                register(alias, spec, "alias")
            if spec.model_path:
                register(spec.model_path, spec, "model path")
            if spec.source:
                register(spec.source, spec, "source")
            if spec.hf_repo:
                register(spec.hf_repo, spec, "Hugging Face repository")

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().lower().replace("_", "-").replace(".", "-")

    @classmethod
    def from_toml_file(cls, config_file: str | os.PathLike[str]) -> ModelCatalog:
        path = Path(config_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"configuration file does not exist: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Extract common/default properties
        common: dict[str, Any] = {}
        for top_k, top_v in data.items():
            if not isinstance(top_v, dict):
                common[top_k] = top_v

        default_section = data.get("default") or data.get("global") or data.get("*")
        if isinstance(default_section, dict):
            common.update(default_section)

        ignored_common_keys = {
            "name",
            "model_path",
            "model",
            "path",
            "hf_repo",
            "hf-repo",
            "hf_file",
            "hf-file",
            "repo_id",
            "default",
            "is_default",
            "aliases",
            "alias",
        }
        common_without_default = {k: v for k, v in common.items() if k not in ignored_common_keys}

        # Collect model dictionaries
        model_dicts: list[tuple[str, dict[str, Any]]] = []
        if isinstance(data.get("models"), dict):
            models_map = data["models"]
            has_subdicts = any(isinstance(v, dict) for v in models_map.values())
            if has_subdicts:
                for m_name, m_val in models_map.items():
                    if isinstance(m_val, dict):
                        model_dicts.append((m_name, m_val))
            elif any(k in models_map for k in ("model_path", "model", "path", "hf_repo", "hf-repo", "repo_id")):
                model_name = str(models_map.get("name") or "default")
                model_dicts.append((model_name, models_map))
        elif isinstance(data.get("models"), list):
            for item in data["models"]:
                if isinstance(item, dict):
                    m_name = item.get("name")
                    if m_name:
                        model_dicts.append((str(m_name), item))

        for k, v in data.items():
            if k in {"default", "global", "*", "models"}:
                continue
            if isinstance(v, dict):
                model_dicts.append((k, v))

        if not model_dicts:
            raise ValueError(f"no models defined in {path}")

        specs: list[ModelSpec] = []
        for name, values in model_dicts:
            merged = {**common_without_default, **values}
            gpu_memory_utilization = _first(values, "gpu_memory_utilization", "mem_fraction_static")
            if gpu_memory_utilization is None:
                gpu_memory_utilization = _first(common_without_default, "gpu_memory_utilization", "mem_fraction_static")
            for key, expected_type in (
                ("async_scheduling", bool),
                ("flashinfer_autotune", bool),
                ("enable_prefix_caching", bool),
                ("speculative_method", str),
                ("num_speculative_tokens", int),
            ):
                value = merged.get(key)
                if value is not None and type(value) is not expected_type:
                    raise ValueError(f"model {name!r}: {key} must be a TOML {expected_type.__name__}")
            device_ids = merged.get("device_ids")
            if device_ids is not None and (
                not isinstance(device_ids, list) or any(type(item) is not int for item in device_ids)
            ):
                raise ValueError(f"model {name!r}: device_ids must be a TOML integer array")
            model_name = str(values.get("name") or name)
            model_path = (
                merged.get("model_path")
                or merged.get("model")
                or merged.get("path")
                or merged.get("hf_repo")
                or merged.get("repo_id")
                or ""
            )
            hf_repo = merged.get("hf_repo") or merged.get("hf-repo")
            hf_file = merged.get("hf_file") or merged.get("hf-file")

            if not model_path and not hf_repo:
                raise ValueError(f"model entry {model_name!r} has no model_path or hf_repo")

            raw_aliases = merged.get("aliases") or merged.get("alias") or ()
            if isinstance(raw_aliases, str):
                aliases = tuple(a.strip() for a in raw_aliases.split(",") if a.strip())
            elif isinstance(raw_aliases, (list, tuple)):
                aliases = tuple(str(a).strip() for a in raw_aliases)
            else:
                aliases = ()

            raw_vllm_args = merged.get("vllm_args") or merged.get("extra_args") or merged.get("args") or ()
            if isinstance(raw_vllm_args, str):
                vllm_args = tuple(raw_vllm_args.split())
            elif isinstance(raw_vllm_args, (list, tuple)):
                vllm_args = tuple(str(a) for a in raw_vllm_args)
            else:
                vllm_args = ()

            reasoning = merged.get("reasoning")
            if isinstance(reasoning, bool):
                reasoning = "on" if reasoning else "off"
            elif reasoning is not None:
                reasoning = str(reasoning)

            spec = ModelSpec(
                name=model_name,
                model_path=str(model_path) if model_path else "",
                hf_repo=str(hf_repo) if hf_repo else None,
                hf_file=str(hf_file) if hf_file else None,
                context_length=_int_or_none(_first(merged, "context_length", "ctx_size", "c", "ctx-size")),
                kv_cache_dtype=(
                    _parse_kv_cache_dtype(
                        merged.get("cache-type-k") or merged.get("cache_type_k"),
                        merged.get("cache-type-v") or merged.get("cache_type_v"),
                        merged.get("kv_cache_dtype") or merged.get("kv-cache-dtype"),
                    )
                    or "auto"
                ),
                tensor_parallel_size=_int_or_none(_first(merged, "tensor_parallel_size", "tp_size", "tp")),
                data_parallel_size=_int_or_none(_first(merged, "data_parallel_size", "dp_size", "dp")),
                gpu_memory_utilization=_float_or_none(gpu_memory_utilization),
                device_ids=tuple(device_ids) if device_ids is not None else None,
                async_scheduling=merged.get("async_scheduling"),
                flashinfer_autotune=merged.get("flashinfer_autotune"),
                enable_prefix_caching=merged.get("enable_prefix_caching"),
                speculative_method=merged.get("speculative_method"),
                num_speculative_tokens=merged.get("num_speculative_tokens"),
                max_num_seqs=_int_or_none(_first(merged, "max_num_seqs", "max_running_requests")),
                max_num_batched_tokens=_int_or_none(_first(merged, "max_num_batched_tokens", "chunked_prefill_size")),
                embeddings=_truthy(merged.get("embeddings") or merged.get("embedding") or merged.get("is_embedding")),
                pooling=merged.get("pooling"),
                reasoning=reasoning,
                reasoning_budget=_int_or_none(_first(merged, "reasoning_budget", "reasoning-budget")),
                spec_type=merged.get("spec_type") or merged.get("spec-type"),
                rope_scaling=merged.get("rope_scaling") or merged.get("rope-scaling") or merged.get("rope_type"),
                rope_scale=_float_or_none(_first(merged, "rope_scale", "rope-scale", "factor")),
                yarn_orig_ctx=_int_or_none(
                    _first(merged, "yarn_orig_ctx", "yarn-orig-ctx", "orig-ctx", "original_max_position_embeddings")
                ),
                rope_theta=_float_or_none(_first(merged, "rope_theta", "rope-theta")),
                idle_seconds=_float_or_none(
                    _first(merged, "idle_seconds", "idle-seconds", "sleep_idle_seconds", "sleep-idle-seconds")
                ),
                reasoning_parser=merged.get("reasoning_parser") or merged.get("reasoning-parser"),
                tool_call_parser=merged.get("tool_call_parser") or merged.get("tool-call-parser"),
                chat_template=merged.get("chat_template") or merged.get("chat-template"),
                attention_backend=merged.get("attention_backend") or merged.get("attention-backend"),
                vllm_args=vllm_args,
                is_default=_truthy(merged.get("default") or merged.get("is_default")),
                aliases=aliases,
            )
            specs.append(spec)

        return cls(specs, path)

    @classmethod
    def from_preset_file(cls, preset_file: str | os.PathLike[str]) -> ModelCatalog:
        path = Path(preset_file).expanduser().resolve()
        if path.suffix == ".toml":
            return cls.from_toml_file(path)

        root, sections = _parse_preset_file(path)
        common = sections.get("*", {})
        specs: list[ModelSpec] = []

        for name, values in sections.items():
            if name == "*":
                continue
            merged = {**common, **values}
            hf = merged.get("hf")
            hf_repo = merged.get("hf-repo") or merged.get("hf_repo")
            hf_file = merged.get("hf-file") or merged.get("hf_file")
            if hf and not hf_repo:
                hf_repo, _, quantization = hf.partition(":")
                if quantization and not hf_file:
                    hf_file = quantization if quantization.endswith(".gguf") else None

            model = merged.get("model")
            if not model and not hf_repo:
                raise ValueError(f"model preset {name!r} has no model or hf-repo")

            kv_cache_dtype = (
                _parse_kv_cache_dtype(
                    merged.get("cache-type-k") or merged.get("cache_type_k"),
                    merged.get("cache-type-v") or merged.get("cache_type_v"),
                    merged.get("kv-cache-dtype") or merged.get("kv_cache_dtype"),
                )
                or "auto"
            )
            rope_scaling = merged.get("rope-scaling") or merged.get("rope_scaling")
            rope_scale = _float_or_none(merged.get("rope-scale") or merged.get("rope_scale"))
            yarn_orig_ctx = _int_or_none(
                merged.get("yarn-orig-ctx") or merged.get("yarn_orig_ctx") or merged.get("orig-ctx")
            )
            rope_theta = _float_or_none(merged.get("rope-theta") or merged.get("rope_theta"))
            idle_seconds = _float_or_none(
                merged.get("sleep-idle-seconds")
                or merged.get("sleep_idle_seconds")
                or merged.get("idle-seconds")
                or merged.get("idle_seconds")
            )

            specs.append(
                ModelSpec(
                    name=name,
                    model_path=model or "",
                    hf_repo=hf_repo,
                    hf_file=hf_file,
                    context_length=_int_or_none(merged.get("c") or merged.get("ctx-size")),
                    embeddings=_truthy(merged.get("embeddings") or merged.get("embedding")),
                    pooling=merged.get("pooling"),
                    reasoning=merged.get("reasoning"),
                    reasoning_budget=_int_or_none(merged.get("reasoning-budget")),
                    spec_type=merged.get("spec-type"),
                    kv_cache_dtype=kv_cache_dtype,
                    rope_scaling=rope_scaling,
                    rope_scale=rope_scale,
                    yarn_orig_ctx=yarn_orig_ctx,
                    rope_theta=rope_theta,
                    idle_seconds=idle_seconds,
                    is_default=_truthy(merged.get("default") or merged.get("is_default")),
                )
            )

        del root
        return cls(specs, path)

    @classmethod
    def from_file(cls, config_file: str | os.PathLike[str]) -> ModelCatalog:
        path = Path(config_file).expanduser().resolve()
        if path.suffix == ".ini":
            return cls.from_preset_file(path)
        return cls.from_toml_file(path)

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return self._specs

    @property
    def default(self) -> ModelSpec:
        for spec in self._specs:
            if spec.is_default and not spec.embeddings:
                return spec
        return next((spec for spec in self._specs if not spec.embeddings), self._specs[0])

    def get(self, name: str | None) -> ModelSpec:
        if not name or not str(name).strip():
            return self.default
        name_str = str(name).strip()
        spec = (
            self._by_name.get(name_str)
            or self._by_alias.get(name_str)
            or self._by_normalized.get(self._normalize(name_str))
            or self._by_alias.get(self._normalize(name_str))
        )
        if spec is None:
            available = ", ".join(self._by_name)
            raise KeyError(f"unknown model {name!r}; available models: {available}")
        return spec

    def canonicalize(self, name: str | None) -> str:
        return self.get(name).name

    def public_models(self, loaded_names: str | Iterable[str] | None = None) -> list[dict[str, object]]:
        if isinstance(loaded_names, str):
            loaded = {loaded_names}
        else:
            loaded = set(loaded_names or ())
        return [spec.public_dict(spec.name in loaded) for spec in self._specs]


def resolve_model_path(spec: ModelSpec) -> str:
    """Resolve a model path, downloading from HF only if an explicit single file is specified."""

    target = spec.model_path or spec.source
    if not target:
        raise ValueError(f"model {spec.name!r} has no source or model_path")

    expanded = Path(os.path.expandvars(target)).expanduser()
    if expanded.exists():
        return str(expanded.resolve())

    if target.startswith(("/", "./", "../", "~", "$")) or expanded.is_absolute():
        raise FileNotFoundError(f"local model for {spec.name!r} does not exist: {expanded}")

    if spec.hf_file:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface-hub is required to resolve HF model files; run `uv sync`") from exc
        repo = spec.hf_repo or target
        return hf_hub_download(repo_id=repo, filename=spec.hf_file)

    return target
