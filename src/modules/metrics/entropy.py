import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.enums import AverageMethod

from modules.const import INF, EPS
from modules.logger import log
from modules.metrics.utils import batch_dot

_SUM = 'sum'


class Entropy(Metric):
    full_state_update = False
    is_differentiable = True
    higher_is_better = False

    def __init__(self, normalize: bool = None, average: str = None, **kwargs):
        """

        Args:
            normalize (bool): if True will normalize data
        """
        super(Entropy, self).__init__(**kwargs)
        self.normalize = normalize
        self.average = average
        self.add_state("cumulate_entropy", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_sample", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, mask: Tensor = None):
        if self.normalize is None:
            self.normalize = torch.any(torch.sum(preds, axis=1) != 1)

        batch_entropy = entropy(preds, mask, self.normalize, self.average)
        if self.average == None:
            self.cumulate_entropy += batch_entropy.sum()
            self.n_sample += preds.size(0)

        elif self.average == _SUM:
            self.cumulate_entropy += batch_entropy
            self.n_sample += preds.size(0)

        elif self.average == AverageMethod.MICRO:
            self.cumulate_entropy += batch_entropy
            self.n_sample += 1

    def compute(self):
        return self.cumulate_entropy.float() / (self.n_sample + (self.n_sample == 0))
    

def entropy(preds: Tensor, mask: Tensor = None, normalize: bool = None, average: str = None) -> Tensor:
    """

    Args:
        preds (tensor): batch of vector dim-D (BxD)
        mask (tensor): boolean, True <==> padding.
        normalize (bool): If need to renormalize distribution
        average (str): If none, do not average. If average = 'micro', will take the mean over batch

    Returns:

    """
    if mask is None:
        mask = torch.zeros(preds.shape, dtype=torch.float).type_as(preds)
    else:
        mask = mask.float()

    if normalize:
        preds = torch.softmax(preds - INF * mask, axis=1)

    log_preds = - torch.log((preds == 0) * EPS + preds)
    length = (1 - mask).sum(axis=1)
    log_length = torch.log(length + (length == 1)).float()  # If length == 1 -> give length = 2

    entropies = batch_dot(preds, log_preds) / log_length

    if average == AverageMethod.MICRO:
        entropy = entropies.mean()
    elif average == _SUM:
        entropy = entropies.sum()
    else:
        entropy = entropies

    return entropy
