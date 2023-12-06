import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.enums import AverageMethod

from modules.metrics.utils import batch_dot


class PowerJaccard(Metric):
	# Only for binary class
	# Mostly for attention
	
	full_state_update = False
	is_differentiable = True
	higher_is_better = False
	
	def __init__(self, power:float=1., average:str=None, **kwargs):
		"""
		
		Args:
			power (float): how to make jaccard homogene
			average (str): if None, compute a flatten vector over batch
		"""
		super(PowerJaccard, self).__init__(**kwargs)
		self.p = power
		self.average = average
		
		if average == AverageMethod.SAMPLES:
			self.add_state("cumulative_iou", default=torch.tensor(0.), dist_reduce_fx="sum")
			self.add_state("n_sample", default=torch.tensor(0), dist_reduce_fx="sum")
			
		if average == AverageMethod.NONE:
			self.add_state("intersection", default=torch.tensor(0.), dist_reduce_fx="sum")
			self.add_state("union", default=torch.tensor(0.), dist_reduce_fx="sum")
		
	def update(self, preds: Tensor, target: Tensor):
		target = target.float()
		if self.average == AverageMethod.NONE:
			self.intersection = (preds * target).sum()
			self.union = preds.pow(self.p).sum() + target.pow(self.p).sum() - self.intersection
		
		if self.average == AverageMethod.SAMPLES:
			batch_intersection = batch_dot(preds, target)
			batch_union = preds.pow(self.p).sum(dim=-1) + target.pow(self.p).sum(dim=-1) - batch_intersection
			self.cumulative_iou += (batch_intersection / batch_union).sum()
			self.n_sample += preds.size(0)
	
	def compute(self):
		if self.average is None:
			return self.intersection / self.union
		
		if self.average == 'samples':
			return self.cumulative_iou.float() / self.n_sample
			
			
def power_jaccard(preds: Tensor, target: Tensor=None, power:float=1., average:str=None) -> Tensor:
	"""
	
	"""
	assert average is None or average in ['samples'], f'forbidden value of average'
	p = power
	target = target.float()
	
	if average == 'samples':
		batch_intersection = batch_dot(preds, target)
		batch_union = preds.pow(p).sum(dim=-1) + target.pow(p).sum(dim=-1) - batch_intersection
		return (batch_intersection / batch_union).mean()
	
	intersection = (preds * target).sum()
	union = preds.pow(p).sum() + target.pow(p).sum() - intersection
	return intersection / union
