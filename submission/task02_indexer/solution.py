import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from contextlib import contextmanager
import math
import triton
import triton.language as tl


@triton.jit
def _score_reduce_kernel(
    score_ptr,      # (B*S, H, T) bf16
    w_ptr,          # (B*S, H) bf16
    out_ptr,        # (B*S, T) bf16
    B_S, H, T, seqlen, ratio, apply_mask,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    t0 = tl.program_id(1) * BLOCK_T
    t_idx = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_idx < T

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for h in range(0, H):
        s = tl.load(score_ptr + (row * H + h) * T + t_idx, mask=t_mask, other=0.0).to(tl.float32)
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + row * H + h).to(tl.float32)
        # bf16-round the product to match torch elementwise bf16 mul
        p = (s * w).to(tl.bfloat16).to(tl.float32)
        acc += p
    out = acc.to(tl.bfloat16)

    if apply_mask:
        seq_s = row % seqlen
        thr = (seq_s + 1) // ratio
        neg = tl.full((BLOCK_T,), float("-inf"), tl.bfloat16)
        out = tl.where(t_idx >= thr, neg, out)

    tl.store(out_ptr + row * T + t_idx, out, mask=t_mask)


def _fused_score_reduce(index_score, weights, seqlen, ratio, apply_mask):
    B, S, H, T = index_score.shape
    B_S = B * S
    score_flat = index_score.reshape(B_S, H, T)
    w_flat = weights.reshape(B_S, H)
    out = torch.empty((B_S, T), dtype=torch.bfloat16, device=index_score.device)
    BLOCK_T = 128
    grid = (B_S, triton.cdiv(T, BLOCK_T))
    _score_reduce_kernel[grid](
        score_flat, w_flat, out,
        B_S, H, T, seqlen, ratio, 1 if apply_mask else 0,
        BLOCK_T=BLOCK_T,
    )
    return out.reshape(B, S, T)


@triton.jit
def _fused_score_kernel(
    q_ptr,          # (B, S, H, D) bf16
    kv_ptr,         # (B, T, D) bf16
    w_ptr,          # (B, S, H) bf16
    out_ptr,        # (B, S, T) bf16
    B, S, H, T, D, seqlen, ratio, apply_mask,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, H_C: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    t0 = tl.program_id(2) * BLOCK_T
    row = b * S + s
    t_idx = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_idx < T
    d_idx = tl.arange(0, BLOCK_D)
    h_idx = tl.arange(0, H_C)
    h_mask = h_idx < H

    # load q for this (b,s): (H_C, BLOCK_D)
    q = tl.load(q_ptr + (row * H + h_idx)[:, None] * D + d_idx[None, :],
                mask=h_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    # load kv block: (BLOCK_T, BLOCK_D)
    kv = tl.load(kv_ptr + (b * T + t_idx)[:, None] * D + d_idx[None, :],
                 mask=t_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    # scores (H_C, BLOCK_T) = q @ kv.T  (fp32 accum, round to bf16 to match einsum output)
    sc = tl.dot(q, tl.trans(kv), out_dtype=tl.float32)
    sc = sc.to(tl.bfloat16).to(tl.float32)
    sc = tl.maximum(sc, 0.0)
    w = tl.load(w_ptr + row * H + h_idx, mask=h_mask, other=0.0).to(tl.float32)
    p = (sc * w[:, None]).to(tl.bfloat16).to(tl.float32)   # (H_C, BLOCK_T)
    p = tl.where(h_mask[:, None], p, 0.0)
    acc = tl.sum(p, axis=0)   # (BLOCK_T,)
    out = acc.to(tl.bfloat16)

    if apply_mask:
        thr = (s + 1) // ratio
        neg = tl.full((BLOCK_T,), float("-inf"), tl.bfloat16)
        out = tl.where(t_idx >= thr, neg, out)
    tl.store(out_ptr + row * T + t_idx, out, mask=t_mask)


def _fused_score(q, kv, weights, seqlen, ratio, apply_mask, BLOCK_T=128, num_warps=4):
    # q: (B,S,H,D)  kv: (B,T,D)  weights: (B,S,H)
    B, S, H, D = q.shape
    T = kv.shape[1]
    out = torch.empty((B, S, T), dtype=torch.bfloat16, device=q.device)
    BLOCK_D = triton.next_power_of_2(D)
    H_C = triton.next_power_of_2(H)
    grid = (B, S, triton.cdiv(T, BLOCK_T))
    _fused_score_kernel[grid](
        q, kv, weights, out,
        B, S, H, T, D, seqlen, ratio, 1 if apply_mask else 0,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, H_C=H_C, num_warps=num_warps,
    )
    return out


@triton.jit
def _fused_score_kernel_bs(
    q_ptr,          # (B, S, H, D) bf16
    kv_ptr,         # (B, T, D) bf16
    w_ptr,          # (B, S, H) bf16
    out_ptr,        # (B, S, T) bf16
    B, S, H, T, D, seqlen, ratio, apply_mask,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    s0 = tl.program_id(1) * BLOCK_S
    t0 = tl.program_id(2) * BLOCK_T
    t_idx = t0 + tl.arange(0, BLOCK_T)
    t_mask = t_idx < T
    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D
    s_rel = tl.arange(0, BLOCK_S)
    s_idx = s0 + s_rel
    s_mask = s_idx < S

    # early-skip: if apply_mask and this whole t-block is masked (-inf) for every s in the block
    max_thr = (s0 + BLOCK_S) // ratio          # upper bound of threshold over s-block
    skip = (apply_mask != 0) & (t0 >= max_thr)
    if skip:
        neg = tl.full((BLOCK_S, BLOCK_T), float("-inf"), tl.bfloat16)
        srow = b * S + s_idx
        store_mask = s_mask[:, None] & t_mask[None, :]
        tl.store(out_ptr + srow[:, None] * T + t_idx[None, :], neg, mask=store_mask)
        return

    kv = tl.load(kv_ptr + (b * T + t_idx)[:, None] * D + d_idx[None, :],
                 mask=t_mask[:, None] & d_mask[None, :], other=0.0)   # (BLOCK_T, BLOCK_D)
    kvT = tl.trans(kv)                                                # (BLOCK_D, BLOCK_T)

    acc = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)
    for h in range(0, H):
        q = tl.load(q_ptr + ((b * S + s_idx) * H + h)[:, None] * D + d_idx[None, :],
                    mask=s_mask[:, None] & d_mask[None, :], other=0.0)   # (BLOCK_S, BLOCK_D)
        sc = tl.dot(q, kvT, out_dtype=tl.float32)                        # (BLOCK_S, BLOCK_T)
        sc = sc.to(tl.bfloat16).to(tl.float32)
        sc = tl.maximum(sc, 0.0)
        w = tl.load(w_ptr + (b * S + s_idx) * H + h, mask=s_mask, other=0.0).to(tl.float32)
        p = (sc * w[:, None]).to(tl.bfloat16).to(tl.float32)            # (BLOCK_S, BLOCK_T)
        acc += p
    out = acc.to(tl.bfloat16)

    min_thr = (s0 + 1) // ratio
    need_mask = (apply_mask != 0) & ((t0 + BLOCK_T) > min_thr)
    if need_mask:
        thr = (s_idx + 1) // ratio                                       # (BLOCK_S,)
        neg = tl.full((BLOCK_S, BLOCK_T), float("-inf"), tl.bfloat16)
        out = tl.where(t_idx[None, :] >= thr[:, None], neg, out)
    srow = b * S + s_idx
    store_mask = s_mask[:, None] & t_mask[None, :]
    tl.store(out_ptr + srow[:, None] * T + t_idx[None, :], out, mask=store_mask)


def _fused_score_bs(q, kv, weights, seqlen, ratio, apply_mask, BLOCK_S=16, BLOCK_T=128, num_warps=4, num_stages=2):
    B, S, H, D = q.shape
    T = kv.shape[1]
    out = torch.empty((B, S, T), dtype=torch.bfloat16, device=q.device)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(T, BLOCK_T))
    _fused_score_kernel_bs[grid](
        q, kv, weights, out,
        B, S, H, T, D, seqlen, ratio, 1 if apply_mask else 0,
        BLOCK_S=BLOCK_S, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=num_stages,
    )
    return out


world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16


@contextmanager
def set_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None
    return F.linear(x, weight)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class ModelNew(torch.nn.Module):
    def __init__(self, args: ModelArgs, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis
        self._thr_cache = {}

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        if self.freqs_cis.device != x.device:
            self.freqs_cis = self.freqs_cis.to(x.device)
        if self.kv_cache.device != x.device:
            self.kv_cache = self.kv_cache.to(x.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        apply_mask = (start_pos == 0)
        tlen = end_pos // ratio
        if bsz == self.kv_cache.shape[0] and tlen == self.kv_cache.shape[1]:
            kv = self.kv_cache
        else:
            kv = self.kv_cache[:bsz, :tlen]
        index_score = _fused_score_bs(q, kv, weights, seqlen, ratio, apply_mask)
        dev = index_score.device
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            key = (seqlen, ratio, dev)
            thr = self._thr_cache.get(key)
            if thr is None:
                thr = (torch.arange(1, seqlen + 1, device=dev) // ratio).unsqueeze(1)
                self._thr_cache[key] = thr
            mask = topk_idxs >= thr
            if offset != 0:
                topk_idxs = topk_idxs + offset
            topk_idxs = topk_idxs.masked_fill_(mask, -1)
        else:
            topk_idxs += offset
        return topk_idxs


def _make_args():
    return ModelArgs(
        max_batch_size=8,
        max_seq_len=2600,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32,
    )


def get_inputs():
    args = _make_args()
    batch_size = 8
    seq_len = 2600
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16)
    start_pos = 0
    offset = 0
    return [x, qr, start_pos, offset]


def get_init_inputs():
    args = _make_args()
    compress_ratio = 4
    max_seq_len = args.max_seq_len
    rope_theta = 10000.0
    freqs = 1.0 / (rope_theta ** (torch.arange(0, args.rope_head_dim, 2)[:args.rope_head_dim // 2].float() / args.rope_head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).view(max_seq_len, -1)
    kv_cache = torch.randn(args.max_batch_size, args.max_seq_len // compress_ratio, args.index_head_dim, dtype=torch.bfloat16)
    return [args, freqs_cis, kv_cache, compress_ratio]
