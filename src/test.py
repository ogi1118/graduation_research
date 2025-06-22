import torch

x = torch.Tensor([[2,3,4],[8,9,7]])
a = torch.Tensor([1,2,3])
y = x * a

print(y)