import torch, sys, time
sys.path.insert(0, '/data/code_list/ks/task02_indexer')
import solution as SOL
torch.manual_seed(0)
init = SOL.get_init_inputs()
m = SOL.ModelNew(*init)
dev='cuda'
m.wq_b=m.wq_b.to(dev); m.weights_proj=m.weights_proj.to(dev)
m.freqs_cis=m.freqs_cis.to(dev); m.kv_cache=m.kv_cache.to(dev)
ins = SOL.get_inputs()
x,qr,start_pos,offset = [t.to(dev) if torch.is_tensor(t) else t for t in ins]

def sync(): torch.cuda.synchronize()
def T(fn,n=100):
    for _ in range(20): fn()
    sync(); t=time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter()-t)/n*1000

ratio=m.compress_ratio; rd=m.rope_head_dim; seqlen=x.shape[1]; bsz=x.shape[0]
end_pos=seqlen
freqs_cis=m.freqs_cis[:seqlen]
with torch.no_grad():
    print("wq_b        :", f"{T(lambda: m.wq_b(qr)):.3f}")
    q0=m.wq_b(qr).unflatten(-1,(m.n_local_heads,m.head_dim))
    print("rope        :", f"{T(lambda: SOL.apply_rotary_emb(q0[...,-rd:].clone(),freqs_cis)):.3f}")
    print("weights_proj:", f"{T(lambda: m.weights_proj(x)):.3f}")
    q=q0; weights=m.weights_proj(x)*(m.softmax_scale*m.n_heads**-0.5)
    kv=m.kv_cache[:bsz,:end_pos//ratio]
    print("fused_score :", f"{T(lambda: SOL._fused_score_bs(q,kv,weights,seqlen,ratio,True)):.3f}")
    sc=SOL._fused_score_bs(q,kv,weights,seqlen,ratio,True)
    print("topk        :", f"{T(lambda: sc.topk(128,dim=-1)[1]):.3f}")
    ti=sc.topk(128,dim=-1)[1]
    dev2=sc.device
    def post():
        mask=ti>=torch.arange(1,seqlen+1,device=dev2).unsqueeze(1)//ratio
        return torch.where(mask,-1,ti+offset)
    print("post_mask   :", f"{T(post):.3f}")
    print("FULL fwd    :", f"{T(lambda: m.forward(x,qr,start_pos,offset)):.3f}")
