import torch, time
import solution as S

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
    score = S._fused_score_bs(q, kv, weights, seqlen, ratio, True)  # (B,S,T) bf16

    B, Sq, T = score.shape
    K = 128
    idx = score.topk(K, dim=-1)[1]  # reference indices

    # How many rows have ties at the K-th boundary value (bf16 → many equal values)?
    vals = score.topk(K, dim=-1)[0]
    kth = vals[..., -1:]                      # K-th value per row
    # count elements equal to kth value in the full row (ties compete for last slots)
    eq_kth = (score == kth).sum(dim=-1)       # (B,S)
    ties = (eq_kth > 1).sum().item()
    total = B * Sq
    print(f"rows={total}, rows_with_boundary_ties={ties} ({100*ties/total:.1f}%)")
    # distribution of how many valid (non -inf) entries per row
    valid = (score > -1e30).sum(dim=-1)
    print(f"valid-per-row: min={valid.min().item()} max={valid.max().item()} median={valid.median().item()}")
    print(f"rows with valid<K (topk includes -inf padding): {(valid<K).sum().item()} ({100*(valid<K).sum().item()/total:.1f}%)")

    # timing torch.topk isolated
    def tt():
        return score.topk(K, dim=-1)[1]
    for _ in range(20): tt()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(200): tt()
    torch.cuda.synchronize()
    print(f"torch.topk: {(time.perf_counter()-t)/200*1000:.3f} ms")
