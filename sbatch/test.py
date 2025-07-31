import torch, ctypes, gc

print("torch.version.cuda =", torch.version.cuda)
x = torch.randn(1, device="cuda")          # 一定要成功跑到这里
torch.cuda._lazy_init()                    # 再保险一次
torch.cuda.synchronize()

print("CUDA device name :", torch.cuda.get_device_name())
print("----- loaded libs after first kernel ------")
for obj in gc.get_objects():
    if isinstance(obj, ctypes.CDLL):
        name = obj._name or ""
        if any(k in name for k in ("cublas", "cudnn", "nccl", "libcuda")):
            print(name)