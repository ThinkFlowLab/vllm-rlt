# FlashAttention and PD prerequisites

The default Triton attention backend does not require a FlashAttention package.
Select `--attention-backend flash_attn` only after installing the package for
the selected GPU and verifying it imports in the same Python environment as
PyTorch. The runtime does not silently fall back to Triton when that import
fails.

| Visible GPU | `flash_attn` selection | Extra | Paged KV constraint |
| --- | --- | --- | --- |
| SM8 (Ampere/Ada) | FA2 | `pip install -e '.[flash-attn]'` | `--block-size` must be a multiple of 256 |
| SM9 (Hopper) | FA3 | Install the official FA3 package | Use a package built for the installed PyTorch/CUDA combination |
| SM10 (Blackwell) | FA4 | `pip install -e '.[flash-attn4]'` | Use a compatible FA4 package |

An explicit `flash_attn_2`, `flash_attn_3`, or `flash_attn_4` selects that
generation when supported by the GPU. This project's paged FA4 path also
supports SM9; paged FA4 on SM12 is currently rejected. Check the
[FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention)
for the corresponding CUDA/toolchain requirements. In particular, building FA2
from source requires a CUDA 12 or newer toolkit with `nvcc`; a CUDA-enabled
PyTorch wheel alone does not provide that compiler. A prebuilt wheel must match
the installed PyTorch, CUDA, Python, and C++ ABI, and a successful `pip install`
does not establish that its CUDA extension imports or runs.

PD is a separate optional feature. Install its transport with
`pip install -e '.[pd]'` and check `python -c 'import nixl'`. The real
prefill-to-decode transfer tests need at least **two visible GPUs**; the
multi-worker tests need **four**. Installing NIXL on a single-GPU host enables
dependency and CPU contract checks, but cannot run those transfer tests.
