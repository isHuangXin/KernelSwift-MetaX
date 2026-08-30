import torch, sys, time
sys.path.insert(0, '/data/code_list/ks/task02_indexer')
dev='cuda'
torch.manual_seed(0)
B,S,T,K = 8,2600,650,128
score = torch.randn(B,S,T, dtype=torch.bfloat16, device=dev)
# apply a causal-ish mask to mimic real distribution
thr = (torch.arange(1,S+1,device=dev)//4).clamp(max=T)
tcol = torch.arange(T,device=dev)
mask = tcol[None,:] >= thr[:,None]
score_m = score.clone()
score_m[mask[None].expand(B,-1,-1)] = float('-inf')

def timeit(fn, n=100):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

# baseline bf16 topk
r_bf = score_m.topk(K,dim=-1)[1]
print("bf16 topk    :", f"{timeit(lambda: score_m.topk(K,dim=-1)[1]):.3f} ms")
# fp32 topk
sf = score_m.float()
r_f = sf.topk(K,dim=-1)[1]
print("fp32 topk    :", f"{timeit(lambda: score_m.float().topk(K,dim=-1)[1]):.3f} ms")
print("  fp32 idx == bf16 idx:", torch.equal(r_f, r_bf))
# sort-based
print("bf16 sort    :", f"{timeit(lambda: score_m.sort(dim=-1,descending=True)[1][...,:K]):.3f} ms")
r_s = score_m.sort(dim=-1,descending=True)[1][...,:K]
print("  sort idx == topk idx:", torch.equal(r_s, r_bf))
