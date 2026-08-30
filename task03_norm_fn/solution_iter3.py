import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---- Phase 1: per-row RMS (sqsum) — tiny, M rows ----
@triton.jit
def _rms_kernel(r_ptr, rms_ptr, M, K, eps, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    sqsum = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        r = tl.load(r_ptr + pid_m * K + k_idx, mask=k_mask, other=0.0)
        sqsum += tl.sum(r * r)
    tl.store(rms_ptr + pid_m, tl.math.rsqrt(sqsum / K + eps))


# ---- Phase 2: matmul (M,K)x(K,N) with split-K + atomic, then scale by rms ----
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_K': 1024}, num_warps=4),
        triton.Config({'BLOCK_K': 1024}, num_warps=8),
        triton.Config({'BLOCK_K': 2048}, num_warps=8),
        triton.Config({'BLOCK_K': 2048}, num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _mm_splitk_kernel(
    r_ptr, fn_ptr, rms_ptr, out_ptr,
    M, N, K, SPLIT_K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_m >= M:
        return

    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # each split handles a contiguous chunk of K
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)

    for k0 in range(k_start, k_end, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < k_end
        r = tl.load(r_ptr + pid_m * K + k_idx, mask=k_mask, other=0.0)
        fn_tile = tl.load(
            fn_ptr + n_idx[:, None] * K + k_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0,
        )
        acc += tl.sum(fn_tile * r[None, :], axis=1)

    rms = tl.load(rms_ptr + pid_m)
    out = acc * rms
    if SPLIT_K == 1:
        tl.store(out_ptr + pid_m * N + n_idx, out, mask=n_mask)
    else:
        tl.atomic_add(out_ptr + pid_m * N + n_idx, out, mask=n_mask)


def _norm_fn_triton(residual_2d, fn, eps):
    M, K = residual_2d.shape
    N = fn.shape[0]
    BLOCK_N = triton.next_power_of_2(N)
    SPLIT_K = 8

    rms = torch.empty((M,), dtype=torch.float32, device=residual_2d.device)
    _rms_kernel[(M,)](residual_2d, rms, M, K, eps, BLOCK_K=2048)

    out = torch.zeros((M, N), dtype=torch.float32, device=residual_2d.device)
    grid = (M, SPLIT_K)
    _mm_splitk_kernel[grid](
        residual_2d, fn, rms, out,
        M, N, K, SPLIT_K, BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, residual, mhc_fn, mhc_norm_weight, mhc_norm_eps):
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        n0, n1 = residual.shape[0], residual.shape[1]
        residual = residual.flatten(2, 3).float()
        K = residual.shape[-1]
        N = mhc_fn.shape[0]
        r2d = residual.reshape(-1, K).contiguous()
        fn2d = mhc_fn.contiguous().float()
        out = _norm_fn_triton(r2d, fn2d, mhc_norm_eps)
        return out.view(n0, n1, N)


n1 = 13
mhc_mult = 4
hidden_size = 1280
generate_normw = False


def generate_norm_fn_test_data(n1, mhc_mult, hidden_size, generate_normw):
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float)
        .mul(1 + torch.arange(mhc_mult).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(mhc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    normw = None
    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float)
    return [residual, fn, normw, out_grad, 1e-6]


def get_inputs():
    residual, fn, normw, out_grad, mhc_norm_eps = generate_norm_fn_test_data(
        n1, mhc_mult, hidden_size, generate_normw)
    return [residual, fn, None, mhc_norm_eps]


def get_init_inputs():
    return []
