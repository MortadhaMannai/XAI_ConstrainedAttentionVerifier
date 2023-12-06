import torch

def batch_dot(a, b):
	return torch.bmm(a.unsqueeze(dim=1), b.unsqueeze(dim=2)).squeeze()