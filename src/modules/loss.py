import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss, KLDivLoss

# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
from modules import INF
from modules.logger import log


class IoU(_Loss):
	
	def __init__(self, eps:float=1e-7, normalize=None, **kwargs):
		
		"""Computes the Jaccard loss, a.k.a the IoU loss.
		
		Parameters
		----------
		eps : float, optional, default=1e-7
			added to the denominator for numerical stability.
		normalize : ??
		
		Notes
		----------
		PyTorch optimizers minimize a loss. In this case, we would like to maximize the jaccard loss so return the negated jaccard loss.
		
		"""
		super(IoU, self).__init__(**kwargs)
		self.eps = eps
		self.normalize = normalize
		
	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		"""
		
		Parameters
		----------
		input : torch.Tensor
			shape `[B, C, H, W]`. Corresponds to the raw output or logits of the model
		target : torch.Tensor
			shape `[B, H, W]` or `[B, 1, H, W]`
			
		Returns
		-------
		torch.Tensor
			output tensor of shape `[B, 1]`

		"""
		if self.normalize is None:
			self.normalize = (input.abs() > 1.).any()
			if self.normalize:
				log.warn(f'Parameter $normalize$, initially `None`, is now set to `True`')
		
		if self.normalize:
			input = torch.sigmoid(input)
		
		#input = (input >= self.threshold).type(int)
		intersection = input.dot(target.float())
		union = torch.sum(input) + torch.sum(target) - intersection
		jaccard_index = ((intersection + self.eps) / (union + self.eps)).mean()
		return 1 - jaccard_index


class KLDivLoss(KLDivLoss):
	
	def __init__(self, **kwargs):
		"""Overriding Pytorch's KLDivLoss. This version we fix -inf to a very small number (-1e30) to avoid nan.
		
		Parameters
		----------
		**kwargs : See `KLDivLoss Pytorch <https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html>`
		"""
		super(KLDivLoss, self).__init__(**kwargs)
		
	def forward(self, input: Tensor, target: Tensor) -> Tensor:
		return super(KLDivLoss, self).forward(input.masked_fill(input.isinf(), -INF), target.masked_fill(target.isinf(), -INF))
		

if __name__ == '__main__':
	from torchmetrics import JaccardIndex
	
	custom_iou = IoU(normalize=True)
	m_iou = JaccardIndex(2)
	x = torch.tensor([0.2, 0.3, 1., 0.98, 0.5])
	y = torch.tensor([0, 0, 1, 1, 1])
	print(custom_iou(x, y))
	print(1 - m_iou(x, y))