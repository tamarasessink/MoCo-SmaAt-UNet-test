# Load the pre-trained ResNet-50 model.
# Replace the first layer of the SmaAt-UNet model with the first layer of the ResNet-50 model.
# This is necessary because the ResNet-50 model takes a 3-channel input, while the SmaAt-UNet
# model takes an input with n_channels channels. Replace the last layer of the SmaAt-UNet model
# with a layer that has the same number of output classes as the ResNet-50 model. This is necessary
# because the ResNet-50 model was likely trained for a different classification task than the SmaAt-UNet model.
# Copy the pre-trained ResNet-50 weights to the corresponding layers of the modified SmaAt-UNet
# model. You may need to adjust the weights if the layer shapes do not match exactly.

import torchvision.models as models
from torch import nn
from torchvision.models import resnet


class SmaAt_UNet_ResNet50(nn.Module):
    def __init__(self, n_channels=3, n_classes=1000, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet_ResNet50, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        # load pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Replace first layer with ResNet-50's first layer
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        # Replace last layer with a layer for n_classes output classes
        self.outc = nn.Conv2d(resnet.fc.in_features, n_classes, kernel_size=1)

        # Freeze all layers except for outc
        for name, param in self.named_parameters():
            if name != "outc.weight" and name != "outc.bias":
                param.requires_grad = False

        # Copy pre-trained ResNet-50 weights to corresponding layers
        self.outc.weight.data = resnet.fc.weight.data.view(n_classes, -1, 1, 1)
        self.outc.bias.data = resnet.fc.bias.data


        # define SMaAt-UNet layers
        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self
