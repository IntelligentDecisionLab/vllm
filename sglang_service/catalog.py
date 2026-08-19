from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value.strip())


def _float_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value.strip())


def _parse_kv_cache_dtype(
    cache_k: str | None,
    cache_v: str | None,
    explicit_dtype: str | None,
) -> str | None:
    if explicit_dtype and explicit_dtype.strip():
        return explicit_dtype.strip().lower()
    dtype = (cache_k or cache_v or "").strip().lower()
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
    """Parse llama.cpp's small INI dialect.

    llama.cpp permits ``version = 1`` before the first section and uses ``[*]``
    as the common section.  ``configparser`` does not accept the former, so a
    tiny parser is both more compatible and easier to keep dependency-free.
    """

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
    """One llama.cpp preset exposed by the SGLang gateway."""

    name: str
    hf_repo: str | None = None
    hf_file: str | None = None
    model: str | None = None
    context_length: int | None = None
    embeddings: bool = False
    pooling: str | None = None
    reasoning: str | None = None
    reasoning_budget: int | None = None
    spec_type: str | None = None
    kv_cache_dtype: str | None = None
    rope_scaling: str | None = None
    rope_scale: float | None = None
    yarn_orig_ctx: int | None = None
    rope_theta: float | None = None
    idle_seconds: float | None = None

    @property
    def source(self) -> str:
        if self.model:
            return self.model
        if self.hf_repo and self.hf_file:
            return f"{self.hf_repo}/{self.hf_file}"
        return self.hf_repo or ""

    @property
    def is_gguf(self) -> bool:
        return (self.hf_file or self.model or "").lower().endswith(".gguf")

    def public_dict(self, loaded: bool = False) -> dict[str, object]:
        return {
            "id": self.name,
            "object": "model",
            "owned_by": "local",
            "created": 0,
            "loaded": loaded,
            "metadata": {
                "source": self.source,
                "format": "gguf" if self.is_gguf else "huggingface",
                "embedding": self.embeddings,
                "pooling": self.pooling,
                "reasoning": self.reasoning,
                "reasoning_budget": self.reasoning_budget,
                "speculative_type": self.spec_type,
                "kv_cache_dtype": self.kv_cache_dtype,
                "rope_scaling": self.rope_scaling,
                "rope_scale": self.rope_scale,
                "yarn_orig_ctx": self.yarn_orig_ctx,
            },
        }


class ModelCatalog:
    def __init__(self, specs: Iterable[ModelSpec], preset_file: Path):
        self._specs = tuple(specs)
        if not self._specs:
            raise ValueError(f"no named models found in preset file: {preset_file}")
        self.preset_file = preset_file
        self._by_name = {spec.name: spec for spec in self._specs}
        self._by_normalized = {self._normalize(spec.name): spec for spec in self._specs}

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().lower().replace("_", "-")

    @classmethod
    def from_preset_file(cls, preset_file: str | os.PathLike[str]) -> "ModelCatalog":
        path = Path(preset_file).expanduser().resolve()
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
                # llama.cpp also accepts repository[:quantization] in ``hf``.
                hf_repo, _, quantization = hf.partition(":")
                if quantization and not hf_file:
                    hf_file = quantization if quantization.endswith(".gguf") else None

            model = merged.get("model")
            if not model and not hf_repo:
                raise ValueError(f"model preset {name!r} has no model or hf-repo")

            kv_cache_dtype = _parse_kv_cache_dtype(
                merged.get("cache-type-k") or merged.get("cache_type_k"),
                merged.get("cache-type-v") or merged.get("cache_type_v"),
                merged.get("kv-cache-dtype") or merged.get("kv_cache_dtype"),
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
                    hf_repo=hf_repo,
                    hf_file=hf_file,
                    model=model,
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
                )
            )

        # Keep the file's ordering.  ``version`` is intentionally accepted for
        # compatibility but is not used by this gateway.
        del root
        return cls(specs, path)

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return self._specs

    @property
    def default(self) -> ModelSpec:
        return next((spec for spec in self._specs if not spec.embeddings), self._specs[0])

    def get(self, name: str | None) -> ModelSpec:
        if not name:
            return self.default
        spec = self._by_name.get(name) or self._by_normalized.get(self._normalize(name))
        if spec is None:
            available = ", ".join(self._by_name)
            raise KeyError(f"unknown model {name!r}; available models: {available}")
        return spec

    def public_models(self, loaded_name: str | None = None) -> list[dict[str, object]]:
        return [spec.public_dict(spec.name == loaded_name) for spec in self._specs]


def resolve_model_path(spec: ModelSpec) -> str:
    """Resolve a preset's local model path, downloading from HF only if needed."""

    if spec.model:
        path = Path(os.path.expandvars(spec.model)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"local model for {spec.name!r} does not exist: {path}")
        return str(path.resolve())

    if not spec.hf_repo:
        raise ValueError(f"model preset {spec.name!r} has no source")

    if spec.hf_file:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - only hit in a broken install
            raise RuntimeError(
                "huggingface-hub is required to resolve HF model files; run `uv sync`"
            ) from exc
        return hf_hub_download(repo_id=spec.hf_repo, filename=spec.hf_file)

    return spec.hf_repo
