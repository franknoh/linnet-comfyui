"""The nodes, called as ComfyUI would, on the Linnet examples."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent


def _load_nodes():
    # ComfyUI imports the directory by path; the tests do the same.
    spec = importlib.util.spec_from_file_location("linnet_comfyui_nodes", HERE.parent / "nodes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = _load_nodes()

LINNET = Path(os.environ.get("LINNET_REPO", HERE.parent.parent / "Linnet"))
STDLIB = os.environ.get("LINNET_STD", str(LINNET / "stdlib"))
GPT2 = LINNET / "examples/06-gpt2/gpt2.linnet"
GENERICS = json.dumps({"Vocab": 11, "MaxPositions": 16, "H": 8, "Heads": 2, "Layers": 2, "T": "f32"})


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (LINNET / candidate).exists():
                os.environ["LINNET_BIN"] = str(LINNET / candidate)
                break
    if not GPT2.exists():
        pytest.skip("set LINNET_REPO to a Linnet checkout")
    torch.manual_seed(0)


def test_load_run_and_inspect(tmp_path: Path) -> None:
    (model,) = nodes.LoadModel().load(str(GPT2), GENERICS, "equivalent", "off", "cpu", std_root=STDLIB)
    weights = {
        name.removeprefix("root."): torch.randn(parameter.shape) * 0.3 for name, parameter in model.named_parameters()
    }
    save_file(weights, str(tmp_path / "gpt2.safetensors"))
    (model,) = nodes.LoadModel().load(
        str(GPT2), GENERICS, "equivalent", "source", "cpu", weights=str(tmp_path / "gpt2.safetensors"), std_root=STDLIB
    )
    (tokens,) = nodes.TensorFromList().make("[[1, 4, 7, 2]]", "i32")
    logits, second = nodes.RunEntry().run(model, "forward", tokens)
    assert second is None and logits.shape == (1, 4, 11)
    (ids,) = nodes.Argmax().argmax(logits, -1)
    assert ids.shape == (1, 4) and ids.dtype == torch.int32
    shown = nodes.TensorToString().show(ids, 16)
    assert shown["result"][0].startswith("[1, 4] int32")
    info = nodes.ModelInfo().describe(model)
    assert "Model" in info["result"][0] and "blocks" in info["result"][0]
    (same,) = nodes.ResetState().reset(model)
    assert same is model


def test_run_rejects_a_wrong_shape(tmp_path: Path) -> None:
    (model,) = nodes.LoadModel().load(str(GPT2), GENERICS, "exact", "off", "cpu", std_root=STDLIB)
    (bad,) = nodes.TensorFromList().make("[1, 2, 3]", "i32")  # rank 1, the entry wants [B, S]
    with pytest.raises(Exception, match=r"tokens|rank|shape"):
        nodes.RunEntry().run(model, "forward", bad)


def test_image_round_trip() -> None:
    image = torch.rand(2, 5, 7, 3)
    (tensor,) = nodes.ImageToTensor().convert(image, "NCHW", "f32", "-1..1")
    assert tensor.shape == (2, 3, 5, 7) and tensor.min() >= -1
    (back,) = nodes.TensorToImage().convert(tensor, "NCHW", "-1..1")
    torch.testing.assert_close(back, image, atol=1e-6, rtol=0)


def test_json_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        nodes.LoadModel().load(str(GPT2), "[1, 2]", "exact", "off", "cpu", std_root=STDLIB)
    with pytest.raises(ValueError, match="JSON list"):
        nodes.TensorFromList().make("not json", "i32")


def test_node_tables_are_complete() -> None:
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for cls in nodes.NODE_CLASS_MAPPINGS.values():
        spec = cls.INPUT_TYPES()
        assert "required" in spec and hasattr(cls, cls.FUNCTION)
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
