"""
Benchmark comparison: Cooperative vs Warp-Specialized SM80 MLA kernel.

Usage:
    python tests/bench_compare_sm80.py
"""
import math
import random

import torch
import triton

from flash_mla import get_mla_metadata, flash_mla_with_kvcache


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    return cos_diff


@torch.inference_mode()
def bench_one(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, warp_spec):
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(b, s_q, h_q, d)
    block_size = 32
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def run():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal, warp_spec=warp_spec,
        )

    # Warmup
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    t = triton.testing.do_bench(run)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes_val = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * 2  # bf16
    return t, FLOPS, bytes_val


@torch.inference_mode()
def verify_correctness(b, s_q, mean_sk, h_q, h_kv, d, dv, causal):
    """Verify WS kernel produces correct results."""
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(b, s_q, h_q, d)
    block_size = 32
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    # Cooperative (baseline)
    out_coop, lse_coop = flash_mla_with_kvcache(
        q, blocked_k, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=causal, warp_spec=False,
    )

    # Warp-specialized
    out_ws, lse_ws = flash_mla_with_kvcache(
        q, blocked_k, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=causal, warp_spec=True,
    )

    # Reference
    out_ref = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
    lse_ref = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i]
        O, LSE = scaled_dot_product_attention(
            q[i].transpose(0, 1),
            blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
            blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
            h_q=h_q, h_kv=h_kv, is_causal=causal,
        )
        out_ref[i] = O.transpose(0, 1)
        lse_ref[i] = LSE

    cos_coop = cal_diff(out_coop, out_ref, "coop")
    cos_ws = cal_diff(out_ws, out_ref, "ws")

    return cos_coop, cos_ws


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    d, dv = 576, 512
    causal = True

    # Correctness check
    print("=" * 80)
    print("Correctness Verification (WS vs Reference)")
    print("=" * 80)
    for b in [16]:
        for s in [4096]:
            for h_q in [16]:
                cos_coop, cos_ws = verify_correctness(b, 1, s, h_q, h_kv, d, dv, causal)
                status_coop = "PASS" if cos_coop < 8e-5 else "FAIL"
                status_ws = "PASS" if cos_ws < 8e-5 else "FAIL"
                print(f"  b={b}, sk={s}, h_q={h_q}: "
                      f"coop cos_diff={cos_coop:.2e} [{status_coop}], "
                      f"ws cos_diff={cos_ws:.2e} [{status_ws}]")

    # Performance comparison
    print()
    print("=" * 80)
    print("Performance Comparison: Cooperative vs Warp-Specialized")
    print("=" * 80)
    print(f"{'Config':<45} {'Coop (ms)':>10} {'WS (ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    for b in [128]:
        for s in [4096, 8192]:
            for h_q in [16, 32, 64, 128]:
                for s_q in [1, 2]:
                    config = f"b={b}, s_q={s_q}, sk={s}, h_q={h_q}"
                    t_coop, FLOPS, bytes_val = bench_one(b, s_q, s, h_q, h_kv, d, dv, causal, False)
                    t_ws, _, _ = bench_one(b, s_q, s, h_q, h_kv, d, dv, causal, True)
                    speedup = t_coop / t_ws
                    print(f"  {config:<43} {t_coop:>9.3f}  {t_ws:>9.3f}  {speedup:>9.3f}x")

    print()
    print("Done.")
