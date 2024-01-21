import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from collections import OrderedDict# Diff models
'''
Auto-encoders
FCN
DeepLab family
U-Net Family
'''

class block(nn.Module):
    def __init__(self, ch_in, ch_out, bn):
        super(block, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding="same"),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, padding="same"),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x 
     
class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, bn):
        super(DownSample, self).__init__()
        self.Block = nn.Sequential(
                block(ch_in, ch_out, bn),
                block(ch_out, ch_out, bn))
        self.down_sample = nn.MaxPool2d(2)
        print(f"{ch_in}->{ch_out}->{ch_out}->maxpool")

    def forward(self, x):
        skip_out = self.Block(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpSample(nn.Module):
    def __init__(self, ch_in, ch_out, bn, bilinear):
        super(UpSample, self).__init__()
        if bilinear:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_sample = nn.ConvTranspose2d(ch_in-ch_out, ch_in-ch_out, kernel_size=2, stride=2)        
        
        self.Block = nn.Sequential(
                block(ch_in, ch_out, bn),
                block(ch_out, ch_out, bn))
        print(f"upsample {ch_in}->{ch_out}->{ch_out}")

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.Block(x)


class UNet(nn.Module):
    def __init__(self,in_channel,  n_classes, batchnorm=False,  bilinear=False):
        super(UNet, self).__init__()

        # 3->64->64
        self.down_conv1 = DownSample(in_channel, 64, batchnorm)
        # 64->128->128
        self.down_conv2 = DownSample(64, 128, batchnorm)
        # 128->256->256
        self.down_conv3 = DownSample(128, 256, batchnorm)
        # 256->512->512
        self.down_conv4 = DownSample(256, 512, batchnorm)

        # Bottleneck
        # 512->1024->1024
        self.double_conv = nn.Sequential(
                block(512, 1024, batchnorm),
                block(1024, 1024, batchnorm))
        
        # 1024->512->512
        self.up_conv4 = UpSample(512 + 1024, 512, batchnorm, bilinear)
        # 512->256->256
        self.up_conv3 = UpSample(256 + 512, 256, batchnorm, bilinear)
        # 256->128->128
        self.up_conv2 = UpSample(128 + 256, 128, batchnorm, bilinear)
        #128->64->64
        self.up_conv1 = UpSample(128 + 64, 64, batchnorm, bilinear)
        
        # 64->n_classes
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        # print(x.shape)
        x, skip2_out = self.down_conv2(x)
        # print(x.shape)
        x, skip3_out = self.down_conv3(x)
        # print(x.shape)
        x, skip4_out = self.down_conv4(x)
        # print(x.shape)
        x = self.double_conv(x)
        # print(x.shape)
        x = self.up_conv4(x, skip4_out)
        # print(x.shape)
        x = self.up_conv3(x, skip3_out)
        # print(x.shape)
        x = self.up_conv2(x, skip2_out)
        # print(x.shape)
        x = self.up_conv1(x, skip1_out)
        # print(x.shape)
        x = self.out(x)
        # print(x.shape)
        return x
    

# Should be same as U-Net without Skip Connections?
class AutoEncoder(nn.Module):
    def __init__(self, in_channel,  n_classes, batchnorm=False,  bilinear=False):
        super(AutoEncoder, self).__init__()
        
        # 3->64->64
        self.down_conv1 = DownSample(in_channel, 64, batchnorm)
        # 64->128->128
        self.down_conv2 = DownSample(64, 128, batchnorm)
        # 128->256->256
        self.down_conv3 = DownSample(128, 256, batchnorm)
        # 256->512->512
        self.down_conv4 = DownSample(256, 512, batchnorm)

        # Bottleneck
        # 512->1024->1024
        self.double_conv = nn.Sequential(
                block(512, 1024, batchnorm),
                block(1024, 1024, batchnorm))
        
        # 1024->512->512
        self.up_conv4 = UpSample(512 + 1024, 512, batchnorm, bilinear)
        # 512->256->256
        self.up_conv3 = UpSample(256 + 512, 256, batchnorm, bilinear)
        # 256->128->128
        self.up_conv2 = UpSample(128 + 256, 128, batchnorm, bilinear)
        #128->64->64
        self.up_conv1 = UpSample(128 + 64, 64, batchnorm, bilinear)
        
        # 64->n_classes
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x, _ = self.down_conv1(x)
        # print(x.shape)
        x, _ = self.down_conv2(x)
        # print(x.shape)
        x, _ = self.down_conv3(x)
        # print(x.shape)
        x, _ = self.down_conv4(x)
        # print(x.shape)
        x = self.double_conv(x)
        # print(x.shape)
        x = self.up_conv4(x, 0)
        # print(x.shape)
        x = self.up_conv3(x, 0)
        # print(x.shape)
        x = self.up_conv2(x, 0)
        # print(x.shape)
        x = self.up_conv1(x, 0)
        # print(x.shape)
        x = self.out(x)
        # print(x.shape)
        return x


# pre-trained models using resnet50 or resnet101 backbone 
# DeepLabV3
# FCN 
# UNET (again to compare our trained unet to pre-trained ?)

class DeepLabV3(nn.Module):
    def __init__(self, in_channel, n_classes, backbone = 'resnet50'):
        super(DeepLabV3, self).__init__()

        # backbone selection
        if backbone == 'resnet50':
            self.deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        
        elif backbone == 'resnet101':
            self.deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True) 
        else:
            print("Supported backbones: resnet50, resnet101")

        # adjust input channels to our images
        self.deeplab.backbone.conv1 = nn.Conv2d(
            in_channel, self.deeplab.backbone.conv1.out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        # change output channels to our number of classes
        self.deeplab.classifier[-1] = nn.Conv2d(
            self.deeplab.classifier[-1].in_channels, n_classes,
            kernel_size=1
        )
            
    def forward(self, x):
        return self.deeplab(x)
    


class FCN(nn.Module):
    def __init__(self, in_channel,n_classes, backbone = 'resnet50'):
        super(FCN,self).__init__()

        #backbone selection
        if backbone == 'resnet50':
            self.backbone = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        elif backbone == 'resnet101':
            self.backbone = models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        else:
            print("Supported backbones: resnet50, resnet101")

        # adjust input channels
        self.backbone.backbone.conv1 = nn.Conv2d(
            in_channel, self.backbone.backbone.conv1.out_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # change output channel
        self.backbone.classifier[4] = nn.Conv2d(
            self.backbone.classifier[4].in_channels, n_classes,
            kernel_size=1
        )
    
    def forward(self,x):
        return self.backbone(x)




    


