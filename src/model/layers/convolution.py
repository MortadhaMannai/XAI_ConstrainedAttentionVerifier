from typing import Union, Tuple

import torch
from torch import nn

from model.layers.activation import activate_map


class Conv(nn.Module):
	
	def __init__(self,
	    in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[str, Union[int, Tuple[int, int]]] = 0,
        activation: str or callable = 'relu',
	    **kwargs):
		
		super().__init__()
		
		if isinstance(activation, str):
			activation = activate_map[activation.lower()]
		
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				padding=padding,
				**kwargs
			),
			activation
		)
	
	def forward(self, x):
		return self.conv(x)
	
	
class ConvMultiKernel(nn.Module):
	
	def __init__(self,
	             in_channels: int,
	             out_channels: int,
	             kernels: Union[int, Tuple[int, int]],
	             feature_dim: int,
	             activation: str or callable = 'relu',
	             **kwargs):
		super().__init__()
		
		if isinstance(kernels, int):
			ks = range(1, 1+2*kernels, 2)
		elif isinstance(kernels, tuple or list):
			ks = range(kernels[0], kernels[1], 2)
		else:
			ks = [1, 3, 5] # default
		
		self.convs = nn.ModuleList([
			Conv(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=(k, feature_dim),
				padding=(k//2,0),
				activation=activation,
				**kwargs
			) for k in ks
		])
	
	def forward(self, x):
		
		h_seq = [conv(x) for conv in self.convs]    # n_kernel * (B, C_out, L, 1)
		# h_seq = [self.relu(h) for h in h_seq]       # n_kernel * (B, C_out, L, 1)
		h_seq = [h.squeeze(-1) for h in h_seq]      # n_kernel * (B, C_out, L)
		h_seq = torch.cat(h_seq, 1)                 # (B, n_kernel * C_out, L)
		return h_seq