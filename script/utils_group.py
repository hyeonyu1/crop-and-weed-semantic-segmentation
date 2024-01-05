import matplotlib.pyplot as plt
import numpy as np
# import src.am4ip.dataset as ds
import src.am4ip.models as models
from src.am4ip.trainer import BaselineTrainer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchsummary import summary
import src.am4ip.losses as losses
import src.am4ip.metrics as m
import os

def config(builtin,
            model, 
            in_channel,
            pre_trained,
            loss, 
            train,
            ds,
            weight_path="", 
            lr=0.001, 
            metrics=["Pixel Acc", "Mean IoU", "DICE"], 
            batch_size=32,
            epoch=1,
            optimizer="SGD",
            momentum=0.9,
            batchnorm=False, 
            bilinear=False,):

    config={}
        
    if pre_trained:
        return 
    else:
        if not builtin:
            if model == "U-Net":
                loaded_model = models.UNet(in_channel, 
                                        ds.get_class_number(),
                                        batchnorm,
                                        bilinear)
            elif model == "Auto-Encoder":
                loaded_model = models.AutoEncoder(in_channel,
                                                ds.get_class_number(),
                                                batchnorm,
                                                bilinear)
            else:
                raise ValueError(f"The given model, {model} is not implemented")
        else:
            try:
                loaded_model = model
            except:
                raise ValueError(f"The given model, {model} is not implemented")
               
        print(f"Using model: {model}")
        config["model"] = loaded_model

    # summary(loaded_model.cuda(), (3, 1024, 1024))
    if optimizer not in ["SGD", "Adam"]:
        raise ValueError(f"The given optimizer, {optimizer} is not implemented")
    else:
        print(f"Using optimizer: {optimizer}")
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(loaded_model.parameters(), lr=lr, momentum=momentum)
    config["optimizer"] = optimizer
    
    if train:
        if loss not in ["Pixel-Wise Cross-Entropy", "Focal Loss"]:
            raise ValueError(f"The given loss, {loss} is not implemented")
        else:
            print(f"Using loss: {loss}")
            if loss == "Pixel-Wise Cross-Entropy":
                loss = losses.pixel_wise_cross_entropy
            if loss == "Focal Loss":
                loss = losses.focal_loss

 
        config["trainer"] = BaselineTrainer(model=loaded_model, loss=loss, optimizer=optimizer)
        config["train_loader"] = DataLoader(ds, batch_size=batch_size, shuffle=True)
        config["epoch"] = epoch

    metric = []
    for t in metrics:
        if t not in ["Mean Pixel Acc", "Mean IoU", "DICE"]:
            raise ValueError(f"The given metrics, {metrics} is not implemented")
        else:
            print(f"Using metrics: {t}")
            if t == "DICE":
                metric.append(m.DICE())
            if t == "Mean IoU":
                metric.append(m.MIOU())
            if t == "Mean Pixel Acc":
                metric.append(m.PixACC())

    config["metrics"] = metric

    print("checking")
    if not os.path.exists('./model'):
        os.makedirs("./model")
    if not os.path.exists('./avg_loss'):
        os.makedirs('./avg_loss')
    if not os.path.exists('./metrics'):
        os.makedirs('./metrics')
    if not os.path.exists('./pred'):
        os.makedirs('./pred')
    if not os.path.exists('./loss_curve'):
        os.makedirs('./loss_curve')
    if not os.path.exists('./metrics_curve'):
        os.makedirs('./metrics_curve')
    return config



class EDA():
    #https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection
    id2cls: dict = {0: "background",
                1: "crop",
                2: "weed",
                3: "partial-crop",
                4: "partial-weed"}
    def __init__(self, dataset):
        self.dataset = dataset 
        self.instance = 0


    def getInfo(self, example=True, idx=5):
        print(f'Dataset Type: {self.dataset.set_type}')
        print(f'Number of classes: {self.dataset.get_class_number()}')
        print(f'Length of {self.dataset.set_type} dataset: {self.dataset.__len__()}')
        print(f'Image size: {self.dataset.__getitem__(idx)[0].shape}')
        print(f'Instance counts: {self.get_counts()}')

        if example:
            print(f'showing example')
            self.get_sample(idx)
            print(f'Showing {[self.id2cls[i] for i in self.instance]}')


    def get_counts(self):
        if self.dataset.get_class_number()==5:
            count_dict = {"background": 0,
                    "crop": 0,
                    "weed": 0,
                    "partial-crop": 0,
                    "partial-weed": 0}
        else:
            count_dict = {"background": 0,
                "crop": 0,
                "weed": 0}
        for i in range(0, int(self.dataset.__len__())):
            _, target = self.dataset.__getitem__(i)
            unique_target = np.unique(np.array(target))
            ids = [self.id2cls[i] for i in unique_target]
            for clsid in ids:
                count_dict[clsid]+=1

        cls_name = list(count_dict.keys())
        cls_count = list(count_dict.values())
        plt.bar(cls_name,cls_count)
        plt.show()
        
        return count_dict


    def get_sample(self, idx=5):
        input_img, target = self.dataset.__getitem__(idx)
        self.instance= np.unique(np.array(target))
        plt.figure(figsize=(12,8))
        plt.subplot(221)
        plt.imshow(input_img)
        plt.title('Input Image')
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(np.array(target)*60, cmap="gray")
        plt.title('Label')
        plt.axis('off')
        plt.subplot(223)
        # plt.imshow()
        plt.title('Blended Image later')
        plt.axis('off')
        plt.subplot(224)
        # plt.imshow()
        plt.title('')
        plt.axis('off')
        plt.show()
