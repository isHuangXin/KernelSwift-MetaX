import torch, sys, time, itertools
sys.path.insert(0, '/data/code_list/ks/task02_indexer')
import importlib
import solution as SOL
importlib.reload(SOL)

torch.manual_seed(0)
init = SOL.get_init_inputs()
ins = SOL.get_inputs()
dev='cuda'
def mv(v):
    if torch.is_tensor(v): return v.to(dev)
    if isinstance(v,(list,tuple)): return type(v)(mv(x) for x in v)
    return v
ins=[mv(x) for x in ins]

# reference topk indices for correctness check
import reference as REF
mref = REF.Model(*REF.get_init_inputs())
mref.wq_b=mref.wq_b.to(dev); mref.weights_proj=mref.weights_proj.to(dev)
torch.manual_seed(42)
ref_ins = mv(REF.get_inputs())
# use same seeded inputs for both
def bench_config(BLOCK_T, num_warps):
    m = SOL.ModelNew(*init)
    m.wq_b=m.wq_b.to(dev); m.weights_proj=m.weights_proj.to(dev)
    # monkeypatch default block/warp
    orig = SOL._fused_score
    def patched(q,kv,w,seqlen,ratio,am):
        return orig(q,kv,w,seqlen,ratio,am,BLOCK_T=BLOCK_T,num_warps=num_warps)
    SOL._fused_score = patched
    with torch.no_grad():
        for _ in range(30): m.forward(*ins)
        torch.cuda.synchronize()
        N=100; t=time.perf_counter()
        for _ in range(N): m.forward(*ins)
        torch.cuda.synchronize()
        ms=(time.perf_counter()-t)/N*1000
    SOL._fused_score = orig
    return ms

for BT in [64,128,256]:
    for nw in [2,4,8]:
        try:
            ms=bench_config(BT,nw)
            print(f"BLOCK_T={BT:4d} warps={nw}: {ms:.3f} ms")
        except Exception as e:
            print(f"BLOCK_T={BT:4d} warps={nw}: FAIL {str(e)[:60]}")
