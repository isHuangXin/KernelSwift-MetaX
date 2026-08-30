import torch, sys, time
sys.path.insert(0, '/data/code_list/ks/task02_indexer')
import solution as SOL
torch.manual_seed(0)
init = SOL.get_init_inputs()
m = SOL.ModelNew(*init)
dev='cuda'
m.wq_b=m.wq_b.to(dev); m.weights_proj=m.weights_proj.to(dev)
m.freqs_cis=m.freqs_cis.to(dev); m.kv_cache=m.kv_cache.to(dev)
ins=[t.to(dev) if torch.is_tensor(t) else t for t in SOL.get_inputs()]
x,qr,sp,off=ins
ratio=m.compress_ratio; seqlen=x.shape[1]; bsz=x.shape[0]
q=m.wq_b(qr).unflatten(-1,(m.n_local_heads,m.head_dim))
SOL.apply_rotary_emb(q[...,-m.rope_head_dim:], m.freqs_cis[:seqlen])
weights=m.weights_proj(x)*(m.softmax_scale*m.n_heads**-0.5)
kv=m.kv_cache[:bsz,:seqlen//ratio]
# reference score for correctness (from current kept kernel BLOCK_S=16)
ref=SOL._fused_score_bs(q,kv,weights,seqlen,ratio,True,BLOCK_S=16,BLOCK_T=128,num_warps=4)
def T(fn,n=200):
    for _ in range(30): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
for BS in [8,16,32,64]:
    for BT in [64,128]:
        for nw in [4,8]:
            try:
                out=SOL._fused_score_bs(q,kv,weights,seqlen,ratio,True,BLOCK_S=BS,BLOCK_T=BT,num_warps=nw)
                ok=torch.equal(out,ref)
                ms=T(lambda: SOL._fused_score_bs(q,kv,weights,seqlen,ratio,True,BLOCK_S=BS,BLOCK_T=BT,num_warps=nw))
                print(f"BS={BS:3d} BT={BT:4d} w={nw}: {ms:.3f} ms  eq={ok}")
            except Exception as e:
                print(f"BS={BS:3d} BT={BT:4d} w={nw}: FAIL {str(e)[:50]}")
