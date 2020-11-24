import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

class ProbCatCrossEntropyLoss(_WeightedLoss):
    #takes probability as input
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ProbCatCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        return
    
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-4)
        return F.nll_loss(torch.log(inputs), targets, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
    