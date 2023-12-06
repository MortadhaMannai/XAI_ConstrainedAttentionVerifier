from typing import Optional, List, Any

import torch
from torch import Tensor
from torchmetrics import AUC
from torchmetrics import functional as F


class AURecall(AUC):
	
	is_differentiable: bool = False
	higher_is_better: Optional[bool] = None
	full_state_update: bool = False
	x: List[Tensor]
	y: List[Tensor]
	
	def __init__(self, reorder: bool = True, **kwargs):
		super(AURecall, self).__init__(reorder, **kwargs)
	
	def update(self, preds: Tensor, target: Tensor):
		thresholds = preds.unique()
		precisions = torch.tensor([F.recall(preds, target, threshold=eps) for eps in thresholds], device=self.device)
		super(AURecall, self).update(thresholds, precisions)
		
	