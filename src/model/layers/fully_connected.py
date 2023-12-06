from torch import nn

from model.layers.activation import activate_map


class FullyConnected(nn.Module):
	
	def __init__(self, d_in: int, d_out: int, dropout: float = 0, activation: str or callable = 'relu'):
		super().__init__()
		if isinstance(activation, str):
			activation = activate_map[activation.lower()]
		self.fc = nn.Sequential(
			nn.Linear(d_in, d_out),
			nn.Dropout(p=dropout),
			activation
		)
	
	def forward(self, x):
		return self.fc(x)