import src.am4ip.dataset as ds
import script.utils_group as utils
import numpy as np
from torchvision import transforms
# import torchvisio
from torch.utils.data import DataLoader
import torch
from torchvision.models.segmentation import fcn_resnet50
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from script.noise import AddGaussianNoise, AddSaltandPepperNoise
'''
Need to do!
- data analysis (EDA)
- data preprocess/aug
- Model Arch (src/am4ip/models.py)

'''


def main():
    # option = "EDA"
    option = "Train"
    # option = "Eval"

    if option == "EDA":
        EDA_dataset = ds.CropSegmentationDataset(
                                            transform = np.array,
                                            target_transform = np.array,
                                            merge_small_items=False)
        EDA = utils.EDA(EDA_dataset)
        EDA.getInfo(example=False, idx=5)
        
    # set up configuration
    
    elif option == "Train":
        train = True
    else:
        train = False

    # Create Config
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((512, 512))])
    # target_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((512, 512))])


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

    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 256))])
    # target_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((256, 256))])
    merge_small_items = False
    remove_small_items = True

    batch_size = 10
    epochs = 20
    
    # loss = ["Pixel-Wise Cross-Entropy", "Focal Loss"]
    loss = ["Focal Loss"]

    metrics = ["DICE"]
    # metrics = ["Mean Pixel Acc", "Mean IoU", "DICE"]
    # learing_rate = [0.1, 0.01, 0.001, 0.0001]
    # learing_rate = [1e-1,1e-2,1e-4]
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
    
    file_name = f"aa_{model_name}_{backbone}"
    # model_name =  fcn_resnet50(num_classes=in_channel, progress=False)
    for lr in learing_rate:
        for l in loss:
            name = f"{file_name}_lr_{lr}_loss_{l}"
            conf = utils.config(builtin,
                    model=model_name,
                    in_channel = in_channel,
                    pre_trained=False,
                    loss=l, 
                    train=train,
                    ds=dataset,
                    weight_path="", 
                    lr=lr, 
                    metrics=metrics, 
                    batch_size=batch_size,
                    epoch=epochs,
                    optimizer=optimizer_name,
                    momentum=0.9,
                    batchnorm=True, 
                    bilinear=False,
                    backbone=backbone
                    )

            if train:
                conf["trainer"].fit(train_data_loader=conf["train_loader"], 
                                    val_data_loader=val_dataset,
                                    epoch=conf["epoch"],
                                    metrics=conf["metrics"],
                                    name = f"{name}"
                                    )
                
                # saving curves
                # plt.plot(loss_values)
    
            PATH = f"./model/{name}.pth"
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
            i = random.randint(0,val_loader.__len__())
            input_img, target = val_loader.__getitem__(i)
            pred = model(input_img.unsqueeze(0))
            if isinstance(pred, dict):
                pred = pred["out"]
            # print(input_img.shape)
            # print(pred.shape)
            # print(target.shape)
            # exit(1)
            output = torch.argmax(pred, dim=1)
            output = output.squeeze(0,1)

            input_img = input_img.permute(1, 2, 0)
            target = target.squeeze()

            plt.figure(figsize=(12,8))
            plt.subplot(221)
            plt.imshow(input_img)
            plt.title('Input Image')
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(np.array(target)*60, cmap="gray")
            plt.title('Target')
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(np.array(output)*60, cmap="gray")
            plt.title('Pred')
            plt.axis('off')
            plt.savefig(f'./pred/{name}.png')
            # exit(1)
main()