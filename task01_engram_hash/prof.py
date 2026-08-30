import torch, importlib.util
from torch.profiler import profile, ProfilerActivity
def load(p):
    s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ref=load("reference.py"); sol=load("solution.py")
torch.manual_seed(42); ins=ref.get_inputs(); dev=torch.device("cuda")
ins=[x.to(dev) if isinstance(x,torch.Tensor) else x for x in ins]
mn=sol.ModelNew().to(dev)
for _ in range(50):
    with torch.no_grad(): mn.forward(*ins)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
    for _ in range(100):
        with torch.no_grad(): mn.forward(*ins)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
