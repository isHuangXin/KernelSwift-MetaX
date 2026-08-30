import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_K': 512}, num_warps=8),
        triton.Config({'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_K': 512}, num_warps=2),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def _partial_kernel(
    r_ptr, fn_ptr, part_ptr, ssum_ptr,
    M, N, K, SPLIT_K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    k_stop = tl.minimum(k_start + k_per, K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sq = tl.zeros((), dtype=tl.float32)
    for k0 in range(k_start, k_stop, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < k_stop
        r = tl.load(r_ptr + pid_m * K + k_idx, mask=k_mask, other=0.0).to(tl.float32)
        sq += tl.sum(r * r)
        fn_tile = tl.load(
            fn_ptr + n_idx[:, None] * K + k_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.sum(fn_tile * r[None, :], axis=1)
    # part: (M, SPLIT_K, N), ssum: (M, SPLIT_K)
    tl.store(part_ptr + (pid_m * SPLIT_K + pid_k) * N + n_idx, acc, mask=n_mask)
    tl.store(ssum_ptr + pid_m * SPLIT_K + pid_k, sq)


@triton.jit
def _reduce_kernel(
    part_ptr, ssum_ptr, out_ptr,
    M, N, K, SPLIT_K, eps,
    BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    s_idx = tl.arange(0, BLOCK_S)
    s_mask = s_idx < SPLIT_K

    # sum sqsum over splits
    ss = tl.load(ssum_ptr + pid_m * SPLIT_K + s_idx, mask=s_mask, other=0.0)
    sqsum = tl.sum(ss)
    rms = tl.math.rsqrt(sqsum / K + eps)

    # sum partials over splits: part (M, SPLIT_K, N)
    p = tl.load(
        part_ptr + pid_m * SPLIT_K * N + s_idx[:, None] * N + n_idx[None, :],
        mask=s_mask[:, None] & n_mask[None, :], other=0.0,
    )
    acc = tl.sum(p, axis=0)
    tl.store(out_ptr + pid_m * N + n_idx, acc * rms, mask=n_mask)


def _norm_fn_triton(residual_2d, fn2d, eps):
    M, K = residual_2d.shape
    N = fn2d.shape[0]
    SPLIT_K = 16
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_S = triton.next_power_of_2(SPLIT_K)
    BLOCK_K = 512

    part = torch.empty((M, SPLIT_K, N), dtype=torch.float32, device=residual_2d.device)
    ssum = torch.empty((M, SPLIT_K), dtype=torch.float32, device=residual_2d.device)
    _partial_kernel[(M, SPLIT_K)](
        residual_2d, fn2d, part, ssum, M, N, K, SPLIT_K,
        BLOCK_N=BLOCK_N,
    )
    out = torch.empty((M, N), dtype=torch.float32, device=residual_2d.device)
    _reduce_kernel[(M,)](
        part, ssum, out, M, N, K, SPLIT_K, eps,
        BLOCK_N=BLOCK_N, BLOCK_S=BLOCK_S, num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, residual, mhc_fn, mhc_norm_weight, mhc_norm_eps):
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        n0, n1 = residual.shape[0], residual.shape[1]
        K = residual.shape[2] * residual.shape[3]
        N = mhc_fn.shape[0]
        r2d = residual.reshape(-1, K)
        if not r2d.is_contiguous():
            r2d = r2d.contiguous()
        fn2d = mhc_fn if mhc_fn.is_contiguous() else mhc_fn.contiguous()
        out = _norm_fn_triton(r2d, fn2d, mhc_norm_eps)
        return out.view(n0, n1, N)


n1 = 13
mhc_mult = 4
hidden_size = 1280
generate_normw = False


def generate_norm_fn_test_data(n1, mhc_mult, hidden_size, generate_normw):
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float)
        .mul(1 + torch.arange(mhc_mult).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float)
        * 1e-4 * (1 + torch.arange(mhc_mult).mul(0.01).view(1, -1, 1))
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
