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
    s_false = score.topk(K, dim=-1, sorted=False)[1]
    print("sorted=False idx == sorted=True idx:", torch.equal(s_false, ref))
    def tt(sorted_flag):
        return score.topk(K, dim=-1, sorted=sorted_flag)[1]
    for sf in [True, False]:
        for _ in range(20): tt(sf)
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(200): tt(sf)
        torch.cuda.synchronize()
        print(f"topk sorted={sf}: {(time.perf_counter()-t)/200*1000:.3f} ms")
