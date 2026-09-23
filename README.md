# linnet-comfyui

ComfyUI nodes for [Linnet](https://linnet.franknoh.dev) models: load a
`.linnet` source with a SafeTensors checkpoint, or a model from the
[Nest](https://nest.franknoh.dev) registry, and run its entries on tensors
inside a graph. Shapes and dtypes are checked before anything runs.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/franknoh/linnet-comfyui
pip install -r linnet-comfyui/requirements.txt
```

The nodes need the `linnet` compiler: put the binary on `PATH` or point
`LINNET_BIN` at it, and `LINNET_STD` at the standard library directory
(the `stdlib/` of a Linnet checkout) unless it is installed next to the
binary. Restart ComfyUI; the nodes appear under `linnet`.

## Nodes

| Node | |
| --- | --- |
| Load Linnet model | a `.linnet` file or package, generics as JSON, a SafeTensors path, `numerics`, `compile`, device → `LINNET_MODEL` |
| Load from Nest | a registry name (`gpt2`, `tinyllama-1.1b-chat`) or a local model directory; the checkpoint is downloaded from the Hub |
| Run entry | a model, an entry name, up to four `TENSOR` inputs, entry generics as JSON (`{"Steps": 16}`) → the entry's results |
| Tensor from list | a JSON list and a dtype → `TENSOR`, for token ids |
| Tensor to string | shape, dtype, and values, shown in the node |
| Image to tensor | ComfyUI `IMAGE` (`[B, H, W, C]` float) → `[B, C, H, W]` (or `NHWC`) in a dtype, `0..1` or `-1..1` |
| Tensor to image | the reverse |
| Argmax | token ids from logits |
| Model info | `print(model)`: the block tree with names and shapes |
| Reset state | zeroes `state` members (KV caches) between generations |

A text generation graph: `Load from Nest (gpt2)` → `Run entry (forward)`
with `Tensor from list ([[464, 3139, 286]])` as `input_1` → `Argmax` →
`Tensor to string`. For a decoder with a KV cache, run `decode` with a
`[Batch, 1]` token and a scalar position, and `Reset state` before a new
sequence.

## Development

```bash
pip install -e ".[dev]"
LINNET_BIN=/path/to/linnet LINNET_STD=/path/to/Linnet/stdlib pytest
```

The tests call the node classes directly with the Linnet examples; ComfyUI
itself is not needed.

## License

MIT.
