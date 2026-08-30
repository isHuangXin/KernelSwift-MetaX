import torch, time
import solution as S
m = S.ModelNew(*S.get_init_inputs())
m.wq_b = m.wq_b.cuda()
m.weights_proj = m.weights_proj.cuda()
ins = [t.cuda() if torch.is_tensor(t) else t for t in S.get_inputs()]
with torch.no_grad():
    for _ in range(10):
        o = m.forward(*ins)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        o = m.forward(*ins)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 50 * 1000
print("OK shape", tuple(o.shape), "dtype", o.dtype, "ms", round(ms, 3))
