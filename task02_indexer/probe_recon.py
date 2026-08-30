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
    score = S._fused_score_bs(q, kv, weights, seqlen, ratio, True)
    K = 128
    ref = score.topk(K, dim=-1)[1]

    def reconstruct():
        v, i = score.topk(K, dim=-1, sorted=False)   # unsorted
        # reorder to descending value, tie-break by smaller index (torch stable rule)
        order = torch.argsort(i, dim=-1, stable=True)   # sort by index asc first (stable)
        v2 = torch.gather(v, -1, order); i2 = torch.gather(i, -1, order)
        order2 = torch.argsort(v2, dim=-1, descending=True, stable=True)
        return torch.gather(i2, -1, order2)
    rec = reconstruct()
    print("reconstruct == ref:", torch.equal(rec, ref))
    for name, fn in [("ref", lambda: score.topk(K,dim=-1)[1]), ("reconstruct", reconstruct)]:
        for _ in range(20): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(200): fn()
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter()-t)/200*1000:.3f} ms")
