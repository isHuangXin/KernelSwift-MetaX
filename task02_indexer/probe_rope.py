import torch, time
import solution as S
torch.manual_seed(0)
m = S.ModelNew(*S.get_init_inputs())
m.wq_b = m.wq_b.cuda(); m.weights_proj = m.weights_proj.cuda()
ins = [t.cuda() if torch.is_tensor(t) else t for t in S.get_inputs()]
x, qr, sp, off = ins
seqlen = x.shape[1]; rd = m.rope_head_dim
freqs_cis = m.freqs_cis.cuda()[:seqlen]
with torch.no_grad():
    q0 = m.wq_b(qr).unflatten(-1, (m.n_local_heads, m.head_dim))

    def rope_ref(q):
        qq = q.clone()
        S.apply_rotary_emb(qq[..., -rd:], freqs_cis)
        return qq

    # alternative: real arithmetic. torch complex: (a+bi)(c+di); pairs (x0,x1) with cos/sin
    cos = freqs_cis.real.contiguous()  # (S, rd/2)
    sin = freqs_cis.imag.contiguous()
    def rope_real(q):
        qq = q.clone()
        r = qq[..., -rd:]
        rp = r.float().unflatten(-1, (-1, 2))
        x0 = rp[..., 0]; x1 = rp[..., 1]
        c = cos.view(1, seqlen, 1, rd//2); s = sin.view(1, seqlen, 1, rd//2)
        o0 = x0*c - x1*s; o1 = x0*s + x1*c
        out = torch.stack([o0, o1], dim=-1).flatten(-2).to(q.dtype)
        r.copy_(out)
        return qq

    a = rope_ref(q0); b = rope_real(q0)
    print("rope_real == ref:", torch.equal(a, b))
    for name, fn in [("ref", rope_ref), ("real", rope_real)]:
        for _ in range(20): fn(q0)
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(200): fn(q0)
        torch.cuda.synchronize()
        print(f"{name}: {(time.perf_counter()-t)/200*1000:.3f} ms")
