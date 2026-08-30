import torch, sys, time
sys.path.insert(0, '/data/code_list/ks/task02_indexer')
import reference as R

torch.manual_seed(0)
init = R.get_init_inputs()
m = R.Model(*init)
ins = R.get_inputs()
dev = 'cuda'
# move
def mv(v):
    if torch.is_tensor(v): return v.to(dev)
    if isinstance(v,(list,tuple)): return type(v)(mv(x) for x in v)
    return v
m.wq_b = m.wq_b.to(dev); m.weights_proj = m.weights_proj.to(dev)
ins = [mv(x) for x in ins]

with torch.no_grad():
    for _ in range(20): m.forward(*ins)
    torch.cuda.synchronize()
    # section timing
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(20):
            m.forward(*ins)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
