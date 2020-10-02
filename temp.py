import torch

a = torch.autograd.Variable(torch.tensor([[1, 2], [3., 4]]), requires_grad=True)
b = a ** 2
c = b ** 2
d = b ** 2

x = torch.autograd.grad(d, a)


