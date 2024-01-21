from script.noise import AddGaussianNoise, AddSaltandPepperNoise
import src.am4ip.dataset as ds
import script.utils_group as utils
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    # transforms.RandomCrop((64,64), padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
    # transforms.RandomRotation((1, 359)),
    # transforms.RandomChoice([AddGaussianNoise(0., 0.05),AddSaltandPepperNoise(0.5)])
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

data_loader = ds.CropSegmentationDataset(
                    set_type = "val",
                    transform = data_transform,
                    target_transform = target_transform,
                    merge_small_items=merge_small_items,
                    remove_small_items=remove_small_items)
dataset = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
in_channel = data_loader.__getitem__(0)[0].shape[0]
model_name = "U-Net"
backbone = 'resnet50'
builtin = False
train = False
name = "U-Net_resnet50_lr_0.001_loss_Focal Loss"
PATH = f"./model/{name}.pth"
conf = utils.config(builtin,                   
                    model=model_name,
                    in_channel = in_channel,
                    pre_trained=False,
                    loss="Focal Loss", 
                    train=train,
                    ds=data_loader,
                    weight_path="", 
                    lr=1e-3, 
                    metrics=metrics, 
                    use_cuda_metrics = True,
                    batch_size=batch_size,
                    epoch=epochs,
                    optimizer=optimizer_name,
                    momentum=0.9,
                    batchnorm=True, 
                    bilinear=False,
                    backbone=backbone
                    )
model = conf["model"].to('cuda')
optimizer = conf["optimizer"]
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_info = checkpoint['epoch']        
loss_info = checkpoint['loss']

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

print("\nEvaluation")
model.training = False
scores = torch.zeros(len(conf["metrics"]), dtype=torch.float32)
with torch.no_grad():
    for i, m in enumerate(conf["metrics"]):
        print(f"{m.name} {m.compute_metric(dataset, model)}")



