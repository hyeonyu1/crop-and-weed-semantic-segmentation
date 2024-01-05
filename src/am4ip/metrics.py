import torch
from typing import List
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from torch import Tensor
from .utils import seperateTarget

class Metric(ABC):
    """Abstract metric class to evaluate generative models.
    """
    def __init__(self, name: str, use_cuda: bool = True) -> None:
        self.name = name
        self.use_cuda = use_cuda
        self.epsilon = 1e-6

    def compute_metric(self, data_loader: DataLoader, model: torch.nn.Module) -> torch.Tensor:
        cumulative_score = 0.
        n_batch = 0.
        with torch.no_grad():
            for i, (x, attr) in enumerate(data_loader):
                # Move data to cuda is necessary:
                if self.use_cuda:
                    x = x.cuda()
                    attr = attr.cuda()

                out = self.batch_compute([x, attr], model)
                # print(out)
                cumulative_score += out.sum(dim=0)
                n_batch += 1.

            res = cumulative_score / n_batch
#0.0691
        return res

    @abstractmethod
    def batch_compute(self, inp: List[torch.Tensor], model: torch.nn.Module):
        raise NotImplementedError


# should be mean or no?
class PixACC(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Pixel Accuracy Score", use_cuda=use_cuda)

    def batch_compute(self, inp: List[torch.Tensor], model: torch.nn.Module):
        pred = model(inp[0])
        if isinstance(pred, dict):
            pred=pred["out"]
        batch,classes,h,w = pred.size()

        pred = torch.argmax(pred, dim=1)
        gt = inp[1].squeeze(1)
        mPA = 0

        for b in range(0,batch):
            correct = 0
            total = 0
            pix_acc = 0
            for i in range(0,classes):
                b_mask = gt[b] == i
                correct_prediction_mask = pred[b] == i 
                overlap = torch.logical_and(b_mask, correct_prediction_mask).float().sum()
                correct += overlap
                total += b_mask.sum()
            pix_acc += (correct+self.epsilon)/(total+self.epsilon)
            bPA = pix_acc/classes
        mPA = bPA/batch

        return mPA/classes
    

class MIOU(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Mean Intersection over Union Score", use_cuda=use_cuda)

    def batch_compute(self, inp: List[torch.Tensor], model: torch.nn.Module):     
        pred = model(inp[0])
        if isinstance(pred, dict):
            pred=pred["out"]
        batch,classes,h,w = pred.size()

        pred = torch.argmax(pred, dim=1)
        gt = inp[1].squeeze(1)
        miou = 0

        for b in range(0,batch):
            iou = 0
            for i in range(0,classes):
                b_mask = gt[b] == i
                correct_prediction_mask = pred[b] == i 
                intersection = torch.logical_and(b_mask,correct_prediction_mask).float().sum()
                union = torch.logical_or(b_mask, correct_prediction_mask).float().sum()
                iou += ((intersection + self.epsilon) / (union + self.epsilon))
            biou = iou/classes
        miou = biou/batch
        
        return miou


class DICE(Metric):
    def __init__(self, use_cuda: bool = True):
        super().__init__("Dice Score", use_cuda=use_cuda)

    def batch_compute(self, inp: List[torch.Tensor], model: torch.nn.Module):

        pred = model(inp[0])
        if isinstance(pred, dict):
            pred=pred["out"]


        pred = torch.argmax(pred, dim=1)
        gt = inp[1].squeeze(1)

        dice_score = self.multiclass_dice_coeff(pred, gt, reduce_batch_first=False)
            
        
        return dice_score
    
    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()


    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)




