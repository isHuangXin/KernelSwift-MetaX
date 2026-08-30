import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_K': 256}, num_warps=8),
        triton.Config({'BLOCK_K': 512}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_dot_kernel(
    r_ptr, fn_ptr, out_ptr,
    M, N, K, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_idx = tl.arange(0, BLOCK_M)
    n_idx = tl.arange(0, BLOCK_N)
    m_mask = m_idx < M
    n_mask = n_idx < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sqsum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        # r_tile: (BLOCK_M, BLOCK_K)
        r_tile = tl.load(
            r_ptr + m_idx[:, None] * K + k_idx[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        sqsum += tl.sum(r_tile * r_tile, axis=1)
        # fn_tile: (BLOCK_K, BLOCK_N)  — fn is (N,K), load transposed
        fn_tile = tl.load(
            fn_ptr + n_idx[None, :] * K + k_idx[:, None],
            mask=n_mask[None, :] & k_mask[:, None], other=0.0,
        )
        acc += tl.dot(r_tile, fn_tile)

    rms = tl.math.rsqrt(sqsum / K + eps)          # (BLOCK_M,)
    out = acc * rms[:, None]
    tl.store(
        out_ptr + m_idx[:, None] * N + n_idx[None, :],
        out, mask=m_mask[:, None] & n_mask[None, :],
    )


def _norm_fn_triton(residual_2d, fn, eps):
    M, K = residual_2d.shape
    N = fn.shape[0]
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = triton.next_power_of_2(N)
    out = torch.empty((M, N), dtype=torch.float32, device=residual_2d.device)
    _fused_dot_kernel[(1,)](
        residual_2d, fn, out, M, N, K, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
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
