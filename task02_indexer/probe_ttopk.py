import torch, time, triton
import triton.language as tl
import solution as S

# Custom Triton top-K: per-row iterative selection over T columns.
# torch tie-break: equal values -> smaller index first. We emulate by using a
# composite key = value scaled + (-index) tiny? Risky in bf16. Instead do exact:
# selection sort style is O(K*T) = 128*650 too slow. Use threshold + count approach.
#
# Approach: find the K-th largest value (kth), then output all indices with
# value > kth (guaranteed selected), then fill remaining slots with value == kth
# in ascending index order. This exactly matches torch's stable tie-break.

@triton.jit
def _topk_kernel(score_ptr, out_ptr, R, T, K, BLOCK_T: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    tmask = t < T
    v = tl.load(score_ptr + row * T + t, mask=tmask, other=-float('inf'))
    # binary-search the K-th largest value via counting how many >= mid.
    # values are bf16 loaded as fp32; do count-based selection.
    # Find kth largest: smallest value x such that count(v >= x) >= K but we need exact kth.
    # Simpler: iterative — but here do rank via count of strictly greater.
    # For each element, rank = number of elements strictly greater + number of equal with smaller index.
    # rank < K  => selected. Output position = rank.
    # Compute count_greater per element: sum over j of (v[j] > v[t])
    # and count_equal_smaller: sum over j<t of (v[j]==v[t])
    # This is O(T^2) within one program via broadcasting (650x650 ~ 422k, ok for one row? heavy).
    vj = v[None, :]      # (1, BLOCK_T)
    vi = v[:, None]      # (BLOCK_T, 1)
    greater = (vj > vi) & tmask[None, :]
    cg = tl.sum(greater.to(tl.int32), axis=1)              # count strictly greater
    ti = t[:, None]; tj = t[None, :]
    eq_smaller = (vj == vi) & (tj < ti) & tmask[None, :]
    ce = tl.sum(eq_smaller.to(tl.int32), axis=1)
    rank = cg + ce                                         # 0-based rank, stable tie-break
    sel = (rank < K) & tmask
    # scatter: out[row, rank] = t  for selected
    out_pos = row * K + rank
    tl.store(out_ptr + out_pos, t.to(tl.int64), mask=sel)


def triton_topk(score, K):
    B, Sq, T = score.shape
    R = B * Sq
    sf = score.reshape(R, T).contiguous()
    out = torch.empty((R, K), dtype=torch.int64, device=score.device)
    BLOCK_T = triton.next_power_of_2(T)
    _topk_kernel[(R,)](sf, out, R, T, K, BLOCK_T=BLOCK_T)
    return out.reshape(B, Sq, K)


torch.manual_seed(0)
m = S.ModelNew(*S.get_init_inputs())
m.wq_b = m.wq_b.cuda(); m.weights_proj = m.weights_proj.cuda()
ins = [t.cuda() if torch.is_tensor(t) else t for t in S.get_inputs()]
x, qr, sp, off = ins
ratio = m.compress_ratio; seqlen = x.shape[1]; bsz = x.shape[0]
with torch.no_grad():
    q = m.wq_b(qr).unflatten(-1, (m.n_local_heads, m.head_dim))
    S.apply_rotary_emb(q[..., -m.rope_head_dim:], m.freqs_cis.cuda()[:seqlen])
    weights = m.weights_proj(x) * (m.softmax_scale * m.n_heads ** -0.5)
    kv = m.kv_cache.cuda()[:bsz, :seqlen // ratio]
    score = S._fused_score_bs(q, kv, weights, seqlen, ratio, True)
    K = 128
    ref = score.topk(K, dim=-1)[1]
    mine = triton_topk(score, K)
    print("triton_topk == ref:", torch.equal(mine, ref))
    if not torch.equal(mine, ref):
        diff = (mine != ref).sum().item()
        print("mismatched:", diff, "/", ref.numel())
    for name, fn in [("ref", lambda: score.topk(K,dim=-1)[1]), ("triton", lambda: triton_topk(score,K))]:
        for _ in range(10): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(100): fn()
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter()-t)/100*1000:.3f} ms")
