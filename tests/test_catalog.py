from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from vllm_gateway.catalog import (
    FamilyProfile,
    ModelCatalog,
    ModelSpec,
    find_local_model,
    profile_for,
    scan_local_models,
)

DENSE_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 1024,
    "max_position_embeddings": 40960,
}

HYBRID_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 8,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 4096,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
    },
}


def write_safetensors(path: Path, keys: tuple[str, ...], payload: int = 1000) -> None:
    header = json.dumps({key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for key in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload)


def make_model(
    hub: Path,
    repo_id: str,
    config: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("model.embed_tokens.weight",),
    revision: str = "rev1",
) -> Path:
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    (repo / "blobs").mkdir()
    (snapshot / "config.json").write_text(json.dumps(config))
    write_safetensors(snapshot / "model.safetensors", keys)
    return snapshot


@pytest.fixture
def hub(tmp_path: Path) -> Path:
    path = tmp_path / "hub"
    path.mkdir()
    return path


def test_dense_model_is_listed_with_kv_size_and_parsers(hub: Path) -> None:
    snapshot = make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)

    (info,) = scan_local_models(hub)

    assert info.repo_id == "Qwen/Qwen3-0.6B"
    assert info.snapshot == snapshot.resolve()
    assert info.max_context == 40960
    assert info.kv_bytes_per_token == 2 * 28 * 8 * 128
    assert info.weights_bytes == (snapshot / "model.safetensors").stat().st_size
    assert info.mtp is False
    assert info.profile == FamilyProfile("qwen3", "hermes")


def test_hybrid_model_counts_only_full_attention_layers(hub: Path) -> None:
    make_model(hub, "Qwen/Qwen3.5-9B", HYBRID_CONFIG)

    info = find_local_model(hub, "Qwen/Qwen3.5-9B")

    assert info is not None
    assert info.kv_bytes_per_token == 2 * 2 * 4 * 256
    assert info.profile == FamilyProfile("qwen3", "qwen3_coder")


def test_mtp_requires_weights_and_adds_draft_layer_kv(hub: Path) -> None:
    make_model(hub, "org/with-mtp", HYBRID_CONFIG, keys=("model.layers.0.weight", "mtp.fc.weight"))
    make_model(hub, "org/stripped", HYBRID_CONFIG, keys=("model.layers.0.weight",))

    with_mtp = find_local_model(hub, "org/with-mtp")
    stripped = find_local_model(hub, "org/stripped")

    assert with_mtp is not None and with_mtp.mtp is True
    assert with_mtp.kv_bytes_per_token == 2 * 3 * 4 * 256
    assert stripped is not None and stripped.mtp is False
    assert stripped.kv_bytes_per_token == 2 * 2 * 4 * 256


QWEN35_9B_LIKE: dict[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 32,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 4096,
        "max_position_embeddings": 262144,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
    },
}


def test_hybrid_model_reports_recurrent_state_per_sequence(hub: Path) -> None:
    make_model(hub, "org/hybrid", QWEN35_9B_LIKE, keys=("model.layers.0.weight", "mtp.fc.weight"))
    make_model(hub, "org/dense", DENSE_CONFIG)

    hybrid = find_local_model(hub, "org/hybrid")
    dense = find_local_model(hub, "org/dense")

    ssm = 32 * 128 * 128 * 4
    conv = 3 * (2 * 16 * 128 + 32 * 128) * 2
    assert hybrid is not None
    assert hybrid.linear_state_bytes == 24 * (ssm + conv)
    assert hybrid.kv_bytes_per_token == 2 * (8 + 1) * 4 * 256
    assert dense is not None and dense.linear_state_bytes == 0


def test_layer_count_fallbacks_without_layer_types(hub: Path) -> None:
    interval = {**DENSE_CONFIG, "model_type": "qwen3_next", "full_attention_interval": 4}
    make_model(hub, "org/interval", interval)

    info = find_local_model(hub, "org/interval")

    assert info is not None
    assert info.kv_bytes_per_token == 2 * (28 // 4) * 8 * 128


def test_head_dim_defaults_to_hidden_size_over_heads(hub: Path) -> None:
    config = {key: value for key, value in DENSE_CONFIG.items() if key != "head_dim"}
    make_model(hub, "org/no-head-dim", config)

    info = find_local_model(hub, "org/no-head-dim")

    assert info is not None
    assert info.kv_bytes_per_token == 2 * 28 * 8 * (1024 // 16)


def test_unknown_family_gets_no_parsers() -> None:
    assert profile_for(["llama"]) == FamilyProfile()
    assert profile_for(["qwen3_5", "qwen3_5_text"]) == FamilyProfile("qwen3", "qwen3_coder")
    assert profile_for(["qwen3_moe"]) == FamilyProfile("qwen3", "hermes")


def test_non_language_models_and_incomplete_downloads_are_hidden(hub: Path) -> None:
    make_model(hub, "org/diffusion", {"architectures": ["UNet2DConditionModel"]})

    partial = make_model(hub, "org/partial-blob", DENSE_CONFIG)
    (partial.parent.parent / "blobs" / "abc.incomplete").write_bytes(b"")

    sharded = make_model(hub, "org/missing-shard", DENSE_CONFIG)
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}})
    )
    write_safetensors(sharded / "model-00001.safetensors", ("a",))

    no_weights = make_model(hub, "org/no-weights", DENSE_CONFIG)
    (no_weights / "model.safetensors").unlink()

    unreadable = hub / "models--org--broken"
    (unreadable / "snapshots" / "x").mkdir(parents=True)
    (unreadable / "snapshots" / "x" / "config.json").write_text("{not json")

    make_model(hub, "org/ok", DENSE_CONFIG)
    (hub / "datasets--org--data").mkdir()

    assert [info.repo_id for info in scan_local_models(hub)] == ["org/ok"]


def test_scan_of_missing_cache_is_empty(tmp_path: Path) -> None:
    assert scan_local_models(tmp_path / "missing") == []


def make_catalog(hub: Path, tmp_path: Path) -> ModelCatalog:
    return ModelCatalog.load(tmp_path / "state" / "aliases.json", hub)


def test_alias_is_persisted_and_reloaded(hub: Path, tmp_path: Path) -> None:
    make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)
    catalog = make_catalog(hub, tmp_path)

    spec = catalog.add("small-8k", "Qwen/Qwen3-0.6B", 8192)
    catalog.add("small-32k", "Qwen/Qwen3-0.6B", 32768)

    assert spec.name == "small-8k"
    assert spec.repo_id == "Qwen/Qwen3-0.6B"
    assert spec.context_length == 8192
    assert spec.reasoning_parser == "qwen3"
    assert spec.tool_call_parser == "hermes"
    assert spec.model_path.endswith("snapshots/rev1")

    reloaded = make_catalog(hub, tmp_path)
    assert [item.name for item in reloaded.specs] == ["small-8k", "small-32k"]
    assert reloaded.get("small-32k").context_length == 32768
    assert not list((tmp_path / "state").glob("*.tmp"))


def test_alias_validation(hub: Path, tmp_path: Path) -> None:
    make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)
    catalog = make_catalog(hub, tmp_path)
    catalog.add("ok", "Qwen/Qwen3-0.6B", 4096)

    for alias in ("", "has space", "a/b", "x" * 65):
        with pytest.raises(ValueError, match="alias must match"):
            catalog.add(alias, "Qwen/Qwen3-0.6B", 4096)
    with pytest.raises(ValueError, match="already exists"):
        catalog.add("ok", "Qwen/Qwen3-0.6B", 4096)
    with pytest.raises(KeyError, match="not available in the local cache"):
        catalog.add("other", "org/missing", 4096)
    for context in (512, 40961, "8192", True, None):
        with pytest.raises(ValueError, match="context_length"):
            catalog.add("other", "Qwen/Qwen3-0.6B", context)
    assert [spec.name for spec in catalog.specs] == ["ok"]


def test_alias_lookup_is_exact(hub: Path, tmp_path: Path) -> None:
    make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)
    catalog = make_catalog(hub, tmp_path)
    catalog.add("Small", "Qwen/Qwen3-0.6B", 4096)

    assert catalog.get(" Small ").name == "Small"
    for name in ("small", "Qwen/Qwen3-0.6B", "", None):
        with pytest.raises(KeyError, match="unknown model"):
            catalog.get(name)


def test_remove_alias_updates_state(hub: Path, tmp_path: Path) -> None:
    make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)
    catalog = make_catalog(hub, tmp_path)
    catalog.add("a", "Qwen/Qwen3-0.6B", 4096)
    catalog.add("b", "Qwen/Qwen3-0.6B", 8192)

    assert catalog.remove("a").name == "a"
    with pytest.raises(KeyError):
        catalog.remove("a")

    assert [spec.name for spec in make_catalog(hub, tmp_path).specs] == ["b"]


def test_unavailable_alias_is_kept_in_state_file(hub: Path, tmp_path: Path) -> None:
    state = tmp_path / "state" / "aliases.json"
    state.parent.mkdir()
    state.write_text(
        json.dumps(
            {
                "aliases": {
                    "gone": {"repo_id": "org/gone", "context_length": 4096},
                    "here": {"repo_id": "Qwen/Qwen3-0.6B", "context_length": 4096},
                }
            }
        )
    )
    make_model(hub, "Qwen/Qwen3-0.6B", DENSE_CONFIG)

    catalog = ModelCatalog.load(state, hub)
    assert [spec.name for spec in catalog.specs] == ["here"]
    with pytest.raises(ValueError, match="already exists"):
        catalog.add("gone", "Qwen/Qwen3-0.6B", 4096)
    catalog.add("new", "Qwen/Qwen3-0.6B", 8192)

    assert set(json.loads(state.read_text())["aliases"]) == {"gone", "here", "new"}


def test_corrupt_state_file_fails_startup(hub: Path, tmp_path: Path) -> None:
    state = tmp_path / "aliases.json"
    state.write_text("{broken")

    with pytest.raises(ValueError, match="invalid alias state file"):
        ModelCatalog.load(state, hub)


def test_in_memory_catalog_and_public_models() -> None:
    catalog = ModelCatalog(
        [
            ModelSpec(name="a", model_path="/m/a", repo_id="org/a", context_length=4096),
            ModelSpec(name="b", model_path="/m/b", repo_id="org/b", context_length=8192),
        ]
    )

    listing = catalog.public_models({"b"})

    assert [item["id"] for item in listing] == ["a", "b"]
    assert [item["loaded"] for item in listing] == [False, True]
    assert listing[0]["metadata"] == {"repo_id": "org/a", "context_length": 4096}
    with pytest.raises(ValueError, match="duplicate alias"):
        ModelCatalog([ModelSpec(name="a", model_path="/m"), ModelSpec(name="a", model_path="/m")])
