"""ComfyUI nodes for Linnet models.

Each node is one call on `linnet.torch` or `linnet.nest`. Models travel
between nodes as `LINNET_MODEL` (a `linnet.torch.LinnetModule`), tensors as
`TENSOR` (a `torch.Tensor`), so an entry's inputs and outputs can be wired
to any shape the model declares; the entry's signature is checked before it
runs.
"""

from __future__ import annotations

import json
from typing import Any

import torch

MODEL = "LINNET_MODEL"
TENSOR = "TENSOR"
NUMERICS = ["exact", "equivalent", "fast"]
COMPILE = ["off", "source", "inductor"]
DEVICES = ["auto", "cpu", "cuda"]
DTYPES = ["f32", "bf16", "f16", "i32", "i64", "bool"]

TORCH_DTYPES = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _json_object(text: str, what: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{what} must be a JSON object: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f'{what} must be a JSON object such as {{"H": 64}}')
    return value


def _compile_option(compile: str) -> bool | str:
    return False if compile == "off" else True if compile == "source" else "inductor"


class LoadModel:
    """Materializes a `.linnet` file or package as a PyTorch module."""

    CATEGORY = "linnet"
    RETURN_TYPES = (MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "source": ("STRING", {"default": "", "tooltip": "a .linnet file or a package directory"}),
                "generics": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": 'the root block\'s generics, e.g. {"H": 64, "T": "bf16"}',
                    },
                ),
                "numerics": (NUMERICS, {"default": "equivalent"}),
                "compile": (
                    COMPILE,
                    {
                        "default": "off",
                        "tooltip": "run entries as generated PyTorch source, optionally under torch.compile",
                    },
                ),
                "device": (DEVICES, {"default": "auto"}),
            },
            "optional": {
                "weights": ("STRING", {"default": "", "tooltip": "a .safetensors file or a directory of them"}),
                "bindings": ("STRING", {"default": "", "tooltip": "JSON mapping parameter paths to tensor names"}),
                "root": ("STRING", {"default": ""}),
                "std_root": (
                    "STRING",
                    {"default": "", "tooltip": "the standard library directory (LINNET_STD by default)"},
                ),
            },
        }

    def load(
        self,
        source: str,
        generics: str,
        numerics: str,
        compile: str,
        device: str,
        weights: str = "",
        bindings: str = "",
        root: str = "",
        std_root: str = "",
    ) -> tuple[Any]:
        from linnet.torch import load

        model = load(
            source,
            generics=_json_object(generics, "generics"),
            root=root or None,
            std_root=std_root or None,
            weights=weights or None,
            bindings=bindings or None,
            device=_device(device),
            numerics=numerics,
            compile=_compile_option(compile),
        )
        return (model,)


class LoadFromNest:
    """Loads a model of the Nest registry with its published checkpoint."""

    CATEGORY = "linnet"
    RETURN_TYPES = (MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "name": ("STRING", {"default": "gpt2", "tooltip": "a registry name, or a local model directory"}),
                "numerics": (NUMERICS, {"default": "fast"}),
                "compile": (COMPILE, {"default": "off"}),
                "device": (DEVICES, {"default": "auto"}),
            },
            "optional": {
                "generics": ("STRING", {"default": "{}", "tooltip": "overrides of the card's generics"}),
                "std_root": ("STRING", {"default": ""}),
            },
        }

    def load(
        self, name: str, numerics: str, compile: str, device: str, generics: str = "{}", std_root: str = ""
    ) -> tuple[Any]:
        from linnet import nest

        model = nest.load(
            name,
            backend="torch",
            std_root=std_root or None,
            generics=_json_object(generics, "generics"),
            device=_device(device),
            numerics=numerics,
            compile=_compile_option(compile),
        )
        return (model,)


class RunEntry:
    """Runs one entry of a model on up to four tensors."""

    CATEGORY = "linnet"
    RETURN_TYPES = (TENSOR, TENSOR)
    RETURN_NAMES = ("result", "result_2")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": (MODEL,),
                "entry": ("STRING", {"default": "forward"}),
            },
            "optional": {
                "input_1": (TENSOR,),
                "input_2": (TENSOR,),
                "input_3": (TENSOR,),
                "input_4": (TENSOR,),
                "generics": (
                    "STRING",
                    {"default": "{}", "tooltip": 'entry generics the inputs do not determine, e.g. {"Steps": 16}'},
                ),
            },
        }

    def run(
        self,
        model: Any,
        entry: str,
        input_1: torch.Tensor | None = None,
        input_2: torch.Tensor | None = None,
        input_3: torch.Tensor | None = None,
        input_4: torch.Tensor | None = None,
        generics: str = "{}",
    ) -> tuple[Any, Any]:
        inputs = [t for t in (input_1, input_2, input_3, input_4) if t is not None]
        device = next(model.parameters(), torch.empty(0)).device
        inputs = [t.to(device) for t in inputs]
        with torch.no_grad():
            results = model.run_entry(entry, inputs, generics=_json_object(generics, "generics"))
        if isinstance(results, (tuple, list)):
            first = results[0] if results else None
            second = results[1] if len(results) > 1 else None
            return (first, second)
        return (results, None)


class TensorFromList:
    """A tensor from a JSON list, for token ids and small inputs."""

    CATEGORY = "linnet/tensors"
    RETURN_TYPES = (TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "values": ("STRING", {"default": "[[1, 2, 3]]", "multiline": True}),
                "dtype": (DTYPES, {"default": "i32"}),
            }
        }

    def make(self, values: str, dtype: str) -> tuple[torch.Tensor]:
        try:
            data = json.loads(values)
        except json.JSONDecodeError as error:
            raise ValueError(f"values must be a JSON list: {error}") from None
        return (torch.tensor(data, dtype=TORCH_DTYPES[dtype]),)


class TensorToString:
    """A readable summary of a tensor: shape, dtype, and values (truncated)."""

    CATEGORY = "linnet/tensors"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"tensor": (TENSOR,), "max_elements": ("INT", {"default": 64, "min": 1, "max": 4096})}}

    def show(self, tensor: torch.Tensor, max_elements: int) -> dict[str, Any]:
        flat = tensor.detach().flatten()
        shown = flat[:max_elements].cpu()
        values = shown.float().tolist() if shown.is_floating_point() else shown.tolist()
        suffix = "" if flat.numel() <= max_elements else f" ... ({flat.numel()} elements)"
        text = f"{list(tensor.shape)} {str(tensor.dtype).removeprefix('torch.')}\n{values}{suffix}"
        return {"ui": {"text": [text]}, "result": (text,)}


class ImageToTensor:
    """ComfyUI `IMAGE` (`[B, H, W, C]`, float in 0..1) as a model input."""

    CATEGORY = "linnet/tensors"
    RETURN_TYPES = (TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["NCHW", "NHWC"], {"default": "NCHW"}),
                "dtype": (["f32", "bf16", "f16"], {"default": "f32"}),
                "normalize": (["0..1", "-1..1"], {"default": "0..1"}),
            }
        }

    def convert(self, image: torch.Tensor, layout: str, dtype: str, normalize: str) -> tuple[torch.Tensor]:
        tensor = image
        if normalize == "-1..1":
            tensor = tensor * 2 - 1
        if layout == "NCHW":
            tensor = tensor.permute(0, 3, 1, 2)
        return (tensor.contiguous().to(TORCH_DTYPES[dtype]),)


class TensorToImage:
    """A model output back to ComfyUI `IMAGE`."""

    CATEGORY = "linnet/tensors"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "tensor": (TENSOR,),
                "layout": (["NCHW", "NHWC"], {"default": "NCHW"}),
                "normalize": (["0..1", "-1..1"], {"default": "0..1"}),
            }
        }

    def convert(self, tensor: torch.Tensor, layout: str, normalize: str) -> tuple[torch.Tensor]:
        image = tensor.detach().float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if layout == "NCHW":
            image = image.permute(0, 2, 3, 1)
        if normalize == "-1..1":
            image = (image + 1) / 2
        return (image.clamp(0, 1).contiguous().cpu(),)


class Argmax:
    """Token ids from logits."""

    CATEGORY = "linnet/tensors"
    RETURN_TYPES = (TENSOR,)
    RETURN_NAMES = ("indices",)
    FUNCTION = "argmax"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"tensor": (TENSOR,), "axis": ("INT", {"default": -1, "min": -8, "max": 8})}}

    def argmax(self, tensor: torch.Tensor, axis: int) -> tuple[torch.Tensor]:
        return (tensor.argmax(dim=axis).to(torch.int32),)


class ModelInfo:
    """The model's block tree, as `print(model)` shows it."""

    CATEGORY = "linnet"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "describe"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"model": (MODEL,)}}

    def describe(self, model: Any) -> dict[str, Any]:
        text = str(model)
        return {"ui": {"text": [text]}, "result": (text,)}


class ResetState:
    """Zeroes the model's `state` members (KV caches) so the next run starts fresh."""

    CATEGORY = "linnet"
    RETURN_TYPES = (MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "reset"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"model": (MODEL,)}}

    def reset(self, model: Any) -> tuple[Any]:
        model.reset_state()
        return (model,)


NODE_CLASS_MAPPINGS = {
    "LinnetLoadModel": LoadModel,
    "LinnetLoadFromNest": LoadFromNest,
    "LinnetRunEntry": RunEntry,
    "LinnetTensorFromList": TensorFromList,
    "LinnetTensorToString": TensorToString,
    "LinnetImageToTensor": ImageToTensor,
    "LinnetTensorToImage": TensorToImage,
    "LinnetArgmax": Argmax,
    "LinnetModelInfo": ModelInfo,
    "LinnetResetState": ResetState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinnetLoadModel": "Load Linnet model",
    "LinnetLoadFromNest": "Load from Nest",
    "LinnetRunEntry": "Run entry",
    "LinnetTensorFromList": "Tensor from list",
    "LinnetTensorToString": "Tensor to string",
    "LinnetImageToTensor": "Image to tensor",
    "LinnetTensorToImage": "Tensor to image",
    "LinnetArgmax": "Argmax",
    "LinnetModelInfo": "Model info",
    "LinnetResetState": "Reset state",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
