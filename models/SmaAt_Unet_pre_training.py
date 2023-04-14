import torch.nn as nn
from torch import Tensor

from models.unet_parts import OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM


# TODO: add copyright

class SmaAt_UNet_pre(nn.Module):
    def __init__(self) -> None:
        super(SmaAt_UNet_pre, self).__init__()
        self.n_channels = 3
        self.kernels_per_layer = 2
        self.bilinear = True
        self.reduction_ratio = 16
        self.num_classes = 128

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=self.kernels_per_layer)
        self.down1 = DownDS(64, 128, kernels_per_layer=self.kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=self.kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=self.kernels_per_layer)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=self.kernels_per_layer)
        # (CBAM) are commonly applied to the output of the final convolutional block in an encoder is to emphasize the
        # most discriminative features in the input image that are relevant to the task being solved.
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=self.reduction_ratio)

        # Fully connected layer
        if self.bilinear:
            self.fc = nn.Linear(512, self.num_classes)
        else:
            self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5Att = self.cbam5(x5)

        # reshape to a 1D vector
        x = x5Att.view(x5Att.size(0), -1)

        # fully connected layer
        x = self.fc(x)

        # return the feature vector
        return x
