from skimage.util import random_noise
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., var=0.05):
        self.mean = mean
        self.var = var
        
    def __call__(self, tensor):
        tensor_type = tensor.dtype
        return torch.as_tensor(random_noise(tensor, mode='gaussian', mean=self.mean, var=self.var, clip=True), dtype=tensor_type)

class AddSaltandPepperNoise(object):
    def __init__(self,prob=0.5):
        self.prob = prob

    def __call__(self, tensor):
        tensor_type = tensor.dtype
        return torch.tensor(random_noise(tensor, mode='s&p', salt_vs_pepper=self.prob, clip=True), dtype= tensor_type)
    