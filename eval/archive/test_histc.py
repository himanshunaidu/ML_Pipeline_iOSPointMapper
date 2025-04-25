import torch

t = torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
print(torch.histc(t, bins=1, min=1, max=2))