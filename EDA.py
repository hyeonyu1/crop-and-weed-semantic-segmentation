import src.am4ip.dataset as ds
import script.utils_group as utils
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from torchvision import transforms
# import torchvisio
from torch.utils.data import DataLoader
import torch
from torchvision.models.segmentation import fcn_resnet50
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from skimage.util import random_noise
EDA_dataset = ds.CropSegmentationDataset(
                                            transform = np.array,
                                            target_transform = np.array,
                                            merge_small_items=False)
# EDA = utils.EDA(EDA_dataset)

# # EDA_dataset.
# folder_name = './EDA'
# if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

plt.figure(figsize=(12, 12))
n_example = 0
plot_idx = 1

# while n_example < 20:
#     print(n_example,plot_idx)
#     idx=random.randint(0, EDA_dataset.__len__())
#     input_img, target = EDA_dataset.__getitem__(idx)
#     ax = plt.subplot(5, 4, plot_idx)
#     ax.imshow(input_img)
#     ax.set_aspect('equal')
#     ax.axis('off')
#     ax.set_aspect('equal')
#     plot_idx+=1
#     n_example+=1
# plt.subplots_adjust(wspace=0, hspace=0)

# plt.savefig(f'imgs/dataset_ex.png')


# # plt.savefig(f'{folder_name}/EDA.png')

# EDA_dataset = ds.CropSegmentationDataset(
#                                     set_type = "train",
#                                     transform = np.array,
#                                     target_transform = np.array,
#                                     merge_small_items=False,
#                                     remove_small_items=True)
# EDA = utils.EDA(EDA_dataset)
# EDA.getInfo(example=False, idx=5)

# EDA_dataset = ds.CropSegmentationDataset(
#                                     set_type = "val",
#                                     transform = np.array,
#                                     target_transform = np.array,
#                                     merge_small_items=True,
#                                     remove_small_items=False)
# EDA = utils.EDA(EDA_dataset)
# EDA.getInfo(example=False, idx=5)





class AddGaussianNoise(object):
    def __init__(self, mean=0., var=0.05):
        self.mean = mean
        self.var = var
        
    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='gaussian', mean=self.mean, var=self.var, clip=True))

class AddSaltandPepperNoise(object):
    def __init__(self,prob=0.5):
        self.prob = prob

    def __call__(self, tensor):
        return torch.tensor(random_noise(tensor, mode='s&p', salt_vs_pepper=self.prob, clip=True))
    
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomChoice([AddGaussianNoise(0., 0.05),AddSaltandPepperNoise(0.5)])

        ])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    ])
merge_small_items = False
remove_small_items = True




val_loader = ds.CropSegmentationDataset(
                    set_type = "val",
                    transform = data_transform,
                    target_transform = target_transform,
                    merge_small_items=merge_small_items,
                    remove_small_items=remove_small_items)
while n_example < 6:
    # print(n_example,plot_idx)
    idx=random.randint(0, val_loader.__len__())
    input_img, target = val_loader.__getitem__(idx)
    # ax = plt.subplot(5, 4, plot_idx)
    # ax.imshow(input_img)
    # ax.set_aspect('equal')
    # ax.axis('off')
    # ax.set_aspect('equal')
    # plot_idx+=1
    n_example+=1
# plt.subplots_adjust(wspace=0, hspace=0)