import torch
import importlib.util
from torch.profiler import profile, ProfilerActivity

def load(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base = "/data/code_list/ks/task03_norm_fn"
ref = load(f"{base}/reference.py")
sol = load(f"{base}/solution.py")

torch.manual_seed(42)
inputs = ref.get_inputs()
dev = torch.device("cuda")
inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inputs]

model = ref.Model().to(dev)
modn = sol.ModelNew().to(dev)

# warmup (triggers autotune)
for _ in range(50):
    with torch.no_grad():
        o0 = model.forward(*inputs)
        o1 = modn.forward(*inputs)
torch.cuda.synchronize()

def prof(name, fn):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as p:
        for _ in range(100):
            with torch.no_grad():
                fn()
        torch.cuda.synchronize()
    print(f"\n===== {name} =====")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))

prof("REFERENCE (torch)", lambda: model.forward(*inputs))
prof("SOLUTION (triton)", lambda: modn.forward(*inputs))

# report the autotuned best config
try:
    k = sol._norm_fn_kernel
    print("\n[autotune] best configs seen:")
    for key, cfg in k.cache.items():
        print("  key=", key, "->", cfg)
except Exception as e:
    print("autotune cache introspection failed:", e)
