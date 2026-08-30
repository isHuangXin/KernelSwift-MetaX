import torch, importlib.util
def load(p):
    s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ref=load("reference.py"); sol=load("solution.py")
torch.manual_seed(42); ins=ref.get_inputs(); dev=torch.device("cuda")
ins=[x.to(dev) if isinstance(x,torch.Tensor) else x for x in ins]
mn=sol.ModelNew().to(dev)
for _ in range(50):
    with torch.no_grad(): mn.forward(*ins)
torch.cuda.synchronize()
for _ in range(100):
    with torch.no_grad(): mn.forward(*ins)
torch.cuda.synchronize(); print("done")
