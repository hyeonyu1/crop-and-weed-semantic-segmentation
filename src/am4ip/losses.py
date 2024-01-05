
import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
import torch.nn as nn  
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import seperateTarget
 
    
'''
Loss Functions
- Pixel-wise cross-entropy
- Focal loss
'''

def pixel_wise_cross_entropy(input, target, use_cuda):

    return nn.CrossEntropyLoss()(input, seperateTarget(input, target,use_cuda))


def focal_loss(input, target, use_cuda):
    gamma=2
    CE = F.cross_entropy(input, seperateTarget(input, target,use_cuda))
    pt = torch.exp(-CE)
    loss = -((1-pt)**gamma)*(-1*CE)

    return loss