from typing import Optional, List, Any

import torch
from torch import Tensor
from torchmetrics import AUC
from torchmetrics import functional as F


class AUPrecision(AUC):
	
	is_differentiable: bool = False
	higher_is_better: Optional[bool] = None
	full_state_update: bool = False
	x: List[Tensor]
	y: List[Tensor]
	
	def __init__(self, reorder: bool = True, **kwargs) -> None:
		super(AUPrecision, self).__init__(reorder, **kwargs)
	
	def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
		"""Update state with predictions and targets.

		Args:
			preds: Predictions from model (probabilities, or labels)
			target: Ground truth labels
		"""
		thresholds = preds.unique()
		precisions = torch.tensor([F.precision(preds, target, threshold=eps) for eps in thresholds], device=self.device)
		super(AUPrecision, self).update(thresholds, precisions)
		
	# 09  901