import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_atomic_kernel(
    r_ptr, fn_ptr, out_ptr, rms_ptr,
    M, N, K, SPLIT_K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid=(M, SPLIT_K). Each program: partial dot over its K-chunk -> atomic into out,
    # partial sqsum -> atomic into rms buffer. A separate finalize kernel applies rsqrt+scale.
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
    tl.atomic_add(out_ptr + pid_m * N + n_idx, acc, mask=n_mask)
    tl.atomic_add(rms_ptr + pid_m, sq)


@triton.jit
def _finalize_kernel(out_ptr, rms_ptr, M, N, K, eps, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N
    rms = tl.math.rsqrt(tl.load(rms_ptr + pid_m) / K + eps)
    v = tl.load(out_ptr + pid_m * N + n_idx, mask=n_mask, other=0.0)
    tl.store(out_ptr + pid_m * N + n_idx, v * rms, mask=n_mask)


def _norm_fn_triton(residual_2d, fn2d, eps):
    M, K = residual_2d.shape
    N = fn2d.shape[0]
    SPLIT_K = 16
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = 512

    out = torch.zeros((M, N), dtype=torch.float32, device=residual_2d.device)
    rms = torch.zeros((M,), dtype=torch.float32, device=residual_2d.device)
    _fused_atomic_kernel[(M, SPLIT_K)](
        residual_2d, fn2d, out, rms, M, N, K, SPLIT_K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4,
    )
    _finalize_kernel[(M,)](out, rms, M, N, K, eps, BLOCK_N=BLOCK_N, num_warps=2)
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
