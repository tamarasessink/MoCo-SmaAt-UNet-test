import torch.nn as nn
from torch import Tensor
import torch

from models.unet_parts import OutConv, OutFC
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM


# TODO: add copyright

class SmaAt_UNet_pre(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet_pre, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        # (CBAM) are commonly applied to the output of the final convolutional block in an encoder is to emphasize the
        # most discriminative features in the input image that are relevant to the task being solved.
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # reduce dimensionality
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        if self.bilinear:
            self.fc = nn.Linear(512, self.n_classes)
        else:
            self.fc = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.cbam5(x)

        # average pooling to get the correct output tensor size
        #x = self.adaptive_pool(x)

        # reshape to a 1D vector
        # x = x5Att.view(x5Att.size(0), -1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc(x)

        # return the feature vector
        return x



