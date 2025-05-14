# FlashMLA

***Adapted from：*** https://github.com/deepseek-ai/FlashMLA/

FlashMLA was initially developed based on Hopper(can refer to:https://github.com/deepseek-ai/FlashMLA/), and I adapted it to Ampere GPUs. Due to the different architectures, the performance of Ampere is currently poor due to register overflow. Welcome to add good optimization ideas.

Currently released:
- BF16
- Paged kvcache with block size of 32

## Quick start

### Install

```bash
python setup.py install
```

### Benchmark

```bash
# amphere gpus
python tests/test_flash_mla_sm80.py

# hopper gpus
python tests/test_flash_mla_sm90.py
```

It is able up to 464 GB/s in memory-bound configuration and 59 TFLOPS in computation-bound configuration on A100 SXM, using CUDA 12.8.
For [reference](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), the peak bandwidth and fp16 FLOPS of A100 SXM are 2039 GB/s and 312 TFLOPS respectively. More efforts are needed to optimize the performance.

### Usage

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=True,
    )
    ...
```

## Requirements

- Ampere GPUs
- CUDA 12.3 and above
- PyTorch 2.0 and above

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernel},
      author={Jiashi Li},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```