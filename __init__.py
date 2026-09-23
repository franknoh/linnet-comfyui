"""ComfyUI custom nodes for Linnet models (https://linnet.franknoh.dev)."""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Executed as a plain module rather than ComfyUI's package (the tests
    # collect the directory by path): load the sibling file directly.
    import importlib.util as _util
    from pathlib import Path as _Path

    _spec = _util.spec_from_file_location("linnet_comfyui_nodes", _Path(__file__).with_name("nodes.py"))
    assert _spec is not None and _spec.loader is not None
    _module = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    NODE_CLASS_MAPPINGS = _module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _module.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
