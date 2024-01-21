import src.am4ip.dataset as ds
import script.utils_group as utils
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from skimage.util import random_noise

# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((256, 256))])
# target_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((256, 256))])
# merge_small_items = False
# remove_small_items = True
# EDA_dataset = ds.CropSegmentationDataset(
#                         transform=data_transform,
#                         target_transform=target_transform,
#                         merge_small_items=merge_small_items,
#                         remove_small_items=remove_small_items)
# input_img, target = EDA_dataset.__getitem__(0)

# n_example=0
# while n_example < 5:
   
#     idx=random.randint(0, EDA_dataset.__len__())
#     input_img, target = EDA_dataset.__getitem__(idx)
   
#     n_example+=1


# data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((256, 256))])
# target_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((256, 256))])

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
    
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    # transforms.RandomCrop((64,64), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
    # transforms.RandomRotation((1, 359)),
    transforms.RandomChoice([AddGaussianNoise(0., 0.05),AddSaltandPepperNoise(0.5)])
    ])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    # transforms.RandomCrop((64,64), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
    # transforms.RandomRotation((1, 359))
    ])

merge_small_items = False
remove_small_items = True
batch_size = 10
epochs = 20
loss = ["Focal Loss"]
metrics = ["DICE"]
learing_rate = [1e-3]
optimizer_name = "Adam"
dataset = ds.CropSegmentationDataset(
                    transform=data_transform,
                    target_transform=target_transform,
                    merge_small_items=merge_small_items,
                    remove_small_items=remove_small_items)
val_loader = ds.CropSegmentationDataset(
                    set_type = "val",
                    transform = data_transform,
                    target_transform = target_transform,
                    merge_small_items=merge_small_items,
                    remove_small_items=remove_small_items)
val_dataset = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
in_channel = dataset.__getitem__(0)[0].shape[0]
model_name = "U-Net"
backbone = 'resnet50'
builtin = False
train = False
# name = "U-Net_resnet50_aug_test2_lr_0.001_loss_Focal Loss"
name = "U-Net_resnet50_aug_noise_injection_rotation_crop_v2_lr_0.001_loss_Focal Loss"
PATH = f"./model/{name}.pth"
conf = utils.config(builtin,
                    model=model_name,
                    in_channel = in_channel,
                    pre_trained=False,
                    loss="Focal Loss", 
                    train=train,
                    ds=dataset,
                    weight_path="", 
                    lr=1e-3, 
                    metrics=metrics, 
                    batch_size=batch_size,
                    epoch=epochs,
                    optimizer=optimizer_name,
                    momentum=0.9,
                    batchnorm=True, 
                    bilinear=False,
                    backbone=backbone
                    )
model = conf["model"].to('cpu')
optimizer = conf["optimizer"]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_info = checkpoint['epoch']        
loss_info = checkpoint['loss']

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


model.training = False
model.eval()
n_examples=3
n = 0
plt.figure(figsize=(12, 12))
n_example = 0
plot_idx = 1
img_idx = [5,250,400]
# img_idx = [30,350,405]
while n < n_examples:
    i = img_idx[n]
    input_img, target = val_loader.__getitem__(i)
    pred = model(input_img.unsqueeze(0))
    if isinstance(pred, dict):
        pred = pred["out"]
    output = torch.argmax(pred, dim=1)
    output = output.squeeze(0,1)
    input_img = input_img.permute(1, 2, 0)
    target = target.squeeze()
    ax = plt.subplot(3, 3, plot_idx)
    ax.imshow(input_img)
    ax.set_title('Input Image')
    ax.axis('off')
    plot_idx+=1
    ax = plt.subplot(3, 3, plot_idx)
    ax.imshow(np.array(target)*60, cmap="gray")
    ax.set_title('Target')
    ax.axis('off')
    plot_idx+=1
 
    ax = plt.subplot(3, 3, plot_idx)
    ax.imshow(np.array(output)*60, cmap="gray")
    ax.set_title('Pred')
    ax.axis('off')
    plot_idx+=1
    n+=1

plt.savefig(f'./pred/2_unet_{name}_same.png')
    # exit(1)
    