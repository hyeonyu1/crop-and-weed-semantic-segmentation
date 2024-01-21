# Advanced Methods for Image Processing

## Introduction
This is the code repository for the Image Processing Project Weed and Crop Segmentation.

To configure your environment, please refer to the [Installation](#installation) section.


## Installation
An environment with a pre-configured Python + Torch installation using GPUs is available. Please follow
[this link.](https://dept-info.labri.fr/~mansenca/public/CREMI_deeplearning_env.pdf)


## Checking model
To check the model and running on a private test, there is a eval.py script:
```
python eval.py
```
To change the dataset, please change the set_type in  line 35 in the code to the desired type:
```python
data_loader = ds.CropSegmentationDataset(
                    set_type = "val",
                    transform = data_transform,
                    target_transform = target_transform,
                    merge_small_items=merge_small_items,
                    remove_small_items=remove_small_items)
```
In addition, the data augmentation is commented out:
```python
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
```
The model can be changed in line 46 and 47:
```python
name = "U-Net_resnet50_lr_0.001_loss_Focal Loss"
PATH = f"./model/{name}.pth"
```
U-Net, Auto-Encoder, DeepLabV3, and FCN. Model can be chosen in the corresponsing line 42 and 43: 
```python
model_name = "U-Net"
backbone = 'resnet50'
```
The model_name can be "U-Net", "DeepLabV3", and "FCN".

When the code is run, it will first return the information of the model's state, followed by the optizer's state, and the metric score. 
The best model .pth can be found at https://drive.google.com/file/d/1V458TDD5f98Y5ZWsDzvLdMGJHXHQmuho/view?usp=sharing. Make sure to unzip the file in the base. 
