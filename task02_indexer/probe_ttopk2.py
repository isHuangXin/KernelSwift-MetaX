import torch, time, triton
import triton.language as tl
import solution as S

# O(T log) count-based top-K per row, exact torch stable tie-break.
# Step1: binary search kth largest value (as bf16 bit-pattern order via fp32).
# Step2: emit indices with v > kth (ascending index = torch order among distinct-> but
#   torch orders by value desc; within v>kth still need value-desc order!).
# NOTE: torch returns indices in value-DESC order. Pure count gives rank; rank<K selected,
#   and we scatter t into out[rank] => that reproduces value-desc + stable tie order.
# Compute rank without O(T^2): rank(t) = count(v > v[t]) + count(v==v[t] and j<t).
# Both are per-element reductions but depend on v[t]; can't avoid pairwise easily.
# Alternative that is O(T): sort is O(T log T). Use tl with iterative? Hard.
# Here: do rank via two passes but vectorized over t using cumulative on sorted -- not in triton.
# Fallback: chunk the T^2 into strips to cut register pressure: loop over j-blocks.

@triton.jit
def _topk_kernel(score_ptr, out_ptr, R, T, K, BLOCK_T: tl.constexpr, JB: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    tmask = t < T
    v = tl.load(score_ptr + row * T + t, mask=tmask, other=-float('inf'))
    rank = tl.zeros((BLOCK_T,), dtype=tl.int32)
    for j0 in range(0, T, JB):
        jj = j0 + tl.arange(0, JB)
        jm = jj < T
        vj = tl.load(score_ptr + row * T + jj, mask=jm, other=-float('inf'))
        # for each t, count over this j-block
        gt = (vj[None, :] > v[:, None]) & jm[None, :]
        eq = (vj[None, :] == v[:, None]) & (jj[None, :] < t[:, None]) & jm[None, :]
        rank += tl.sum(gt.to(tl.int32), axis=1) + tl.sum(eq.to(tl.int32), axis=1)
    sel = (rank < K) & tmask
    tl.store(out_ptr + row * K + rank, t.to(tl.int64), mask=sel)


def triton_topk(score, K, JB=128):
    B, Sq, T = score.shape
    R = B * Sq
    sf = score.reshape(R, T).contiguous()
    out = torch.empty((R, K), dtype=torch.int64, device=score.device)
    BLOCK_T = triton.next_power_of_2(T)
    _topk_kernel[(R,)](sf, out, R, T, K, BLOCK_T=BLOCK_T, JB=JB)
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
    print("eq:", torch.equal(mine, ref))
    for name, fn in [("ref", lambda: score.topk(K,dim=-1)[1]), ("triton", lambda: triton_topk(score,K))]:
        for _ in range(10): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(100): fn()
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter()-t)/100*1000:.3f} ms")
