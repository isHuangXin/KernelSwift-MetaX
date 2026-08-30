import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _norm_fn_kernel(
    r_ptr,      # residual, (M, K) fp32
    fn_ptr,     # fn,       (N, K) fp32
    out_ptr,    # output,   (M, N) fp32
    M, N, K,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grid = (M,) — one program per residual row; computes all N outputs for that row.
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    n_idx = tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sqsum = tl.zeros((), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        r = tl.load(r_ptr + pid_m * K + k_idx, mask=k_mask, other=0.0)
        sqsum += tl.sum(r * r)

        fn_tile = tl.load(
            fn_ptr + n_idx[:, None] * K + k_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(fn_tile * r[None, :], axis=1)

    rms = tl.math.rsqrt(sqsum / K + eps)
    out = acc * rms
    tl.store(out_ptr + pid_m * N + n_idx, out, mask=n_mask)


def _norm_fn_triton(residual_2d, fn, eps):
    M, K = residual_2d.shape
    N = fn.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=residual_2d.device)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    _norm_fn_kernel[grid](
        residual_2d, fn, out,
        M, N, K, eps,
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
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
    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float) * 0.1 + 1.0
    else:
        normw = None
    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float)
    return [residual, fn, normw, out_grad, 1e-6]


def get_inputs():
    residual, fn, normw, out_grad, mhc_norm_eps = generate_norm_fn_test_data(
        n1, mhc_mult, hidden_size, generate_normw)
    return [residual, fn, None, mhc_norm_eps]


def get_init_inputs():
    return []
