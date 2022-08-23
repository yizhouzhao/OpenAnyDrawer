import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion import *
from resnet import IdentityBlock, ConvBlock


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),  # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CustomCliport(nn.Module):
    def __init__(self, pred_mode = False, clip_text_feature_path = "text2clip_feature.pickle") -> None:
        """
        Prediction mode: initialize resnet 
        """
        super().__init__()

        self.pred_mode = pred_mode
        self.batchnorm = True
        self.clip_text_feature_path = clip_text_feature_path

        self.proj_input_dim = 512 
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 256)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 128)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 64)

        self.lang_fuser1 = FusionMult(512)
        self.lang_fuser2 = FusionMult(256)
        self.lang_fuser3 = FusionMult(64)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear', align_corners=True),
            DoubleConv(512, 256)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear', align_corners=True),
            DoubleConv(256, 128)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor= 2, mode='bilinear', align_corners=True),
            DoubleConv(128, 64)
        )

        # self.layer1 = nn.Sequential(
        #     ConvBlock(128, [64, 64, 64], kernel_size=3, str ide=1, batchnorm=self.batchnorm),
        #     IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        # )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential( 
            nn.Conv2d(16, 1, kernel_size=1),
        )


        # in prediction
        if self.pred_mode:
            # load vision model
            from transformers import AutoFeatureExtractor, ResNetModel
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
            self.resnet_model = ResNetModel.from_pretrained("microsoft/resnet-18")

            # load language feature
            import pickle
            self.text2clip_feature = pickle.load(open(self.clip_text_feature_path,'rb'))

    def pred_box_pos_and_dir(self, image, text):
        """
        Prediction box center position and direction
        """
        # image features
        inputs = self.feature_extractor(image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = self.resnet_model(**inputs).last_hidden_state # [1, 512, 7, 7]
            text_feautures = self.text2clip_feature[text].unsqueeze(0) # [1, 512]

            pred_y = self.forward(image_features, text_feautures)

        pred_max_index = torch.argmax(pred_y[0].cpu().data).item() 

        h, w =  pred_max_index// 256, pred_max_index % 256

        # get direction
        top_bound = max(h - 5, 0)
        bottom_bound = min(h + 5, 255)

        left_bound = max(w - 5, 0)
        right_bound = min(w + 5, 255)

        # mean over vertical direction
        v_mean = torch.mean(pred_y[0][top_bound:bottom_bound, w]).item()
        h_mean = torch.mean(pred_y[0][left_bound:right_bound, h]).item()

        handle_dir = "horizontal" if v_mean > h_mean else "vertical" # if vertical direction more concentrate, then direciton is horizontal

        return (h,w), handle_dir
        
        
        


        
    def forward(self, x, l):
        """
        x: image features [B x 512 x 7 x 7]
        l: language features [B x 512]
        """
        x = self.up1(x)
        x = self.lang_fuser1(x, l, x2_proj = self.lang_proj1)

        x = self.up2(x)
        x = self.lang_fuser2(x, l, x2_proj = self.lang_proj2)

        x = self.up3(x)
        x = self.lang_fuser3(x, l, x2_proj = self.lang_proj3)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = F.relu(x)
        x = x.squeeze(1)

        return x
