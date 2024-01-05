
from typing import Callable, List
import torch
import torch.utils.data as data
# import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from .metrics import Metric
from .utils import BestModelChecker

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class BaselineTrainer:
    def __init__(self, model: torch.nn.Module,
                 loss: Callable,
                 optimizer: torch.optim.Optimizer,
                 use_cuda=True):
        self.loss = loss
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.best_model_checker = BestModelChecker()
        if use_cuda:
            self.model = model.to(device="cuda:0")

    def fit(self, 
            train_data_loader: data.DataLoader,
            val_data_loader: data.DataLoader,
            epoch: int,
            metrics: List[Metric],
            name: str):
        avg_loss = 0.
        self.model.training = True
        logger = open(f'./avg_loss/{name}.txt', 'a')

        for e in range(epoch):
            print(f"Start epoch {e+1}/{epoch}")
            n_batch = 0
            for i, (ref_img, dist_img) in enumerate(train_data_loader):
                # Reset previous gradients
                self.optimizer.zero_grad()
                # Move data to cuda is necessary:
                if self.use_cuda:
                    ref_img = ref_img.cuda()
                    dist_img = dist_img.cuda().float()
                
                # Make forward
                # TODO change this part to fit your loss function
                output = self.model.forward(ref_img)
                if isinstance(output, dict):
                    output = self.model.forward(ref_img)["out"]

                loss = self.loss(output, dist_img, self.use_cuda)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
                # avg_loss += loss.items()
                avg_loss += loss.item()

                n_batch += 1

                print(f"\r{i+1}/{len(train_data_loader)}: loss = {loss / n_batch}", end='')
            
            
            logger.write(f"Avg loss = {avg_loss/(len(train_data_loader))}\n")
#       
            eval_acc = self.eval(val_data_loader, metrics, name)
            print(f"Metrics: {eval_acc}")

            if len(eval_acc) > 1:
                print(f"Multiple metrics given, using the first one")
                eval_acc = eval_acc[0]
            self.best_model_checker.save(eval_acc, e, self.model, self.optimizer, loss, name)
        
        return avg_loss

    # should save avg loss??
    def eval(self, val_data_loader: data.DataLoader, metrics: List[Metric], name: str) -> torch.Tensor:
        print("\nEvaluation")
        self.model.training = False

        scores = torch.zeros(len(metrics), dtype=torch.float32)
        with torch.no_grad():
            for i, m in enumerate(metrics):
                scores[i] = m.compute_metric(val_data_loader, self.model)
                logger_name = m.name.replace(" ","_")
                logger = open(f'./metrics/{name}_{logger_name}.txt', 'a')
                logger.write(f"{scores[i]}\n")
                print("saved")
        self.model.training = True

        return scores
