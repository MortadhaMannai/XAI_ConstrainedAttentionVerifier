from typing import Optional, List, Any

from torch import Tensor
from torchmetrics import Metric

#TODO: maybe abandone
class AUCModule(Metric):
	is_differentiable: bool = False
	higher_is_better: Optional[bool] = None
	full_state_update: bool = False
	preds: List[Tensor]
	target: List[Tensor]
	
	def __init__(self, reorder: bool = False, **kwargs) -> None:
		super(AUCModule, self).__init__(reorder, **kwargs)
	
	def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
		"""Update state with predictions and targets.

		Args:
			preds: Predictions from model (probabilities, or labels)
			target: Ground truth labels
		"""
		self.preds.append(preds)
		self.target.append(target)
		

