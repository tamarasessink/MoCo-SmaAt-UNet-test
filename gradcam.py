import os

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

from models.SmaAt_Unet_pre_training import SmaAt_UNet_pre
from models.SmaAt_UNet import SmaAt_UNet
from models.unet_precip_regression_lightning import UNetDS_Attention
from utils import dataset_precip


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if device == 'cuda':
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # Move both tensors to the device of self.mask
        model_output = model_output.to(self.mask.device)
        self.mask = self.mask.to(self.mask.device)
        return (model_output[self.category, :, :] * self.mask).sum()


def run_cam(model, target_layers, device, test_dl):
    image_count = 0
    for x, y_true in tqdm(test_dl, leave=False):
        if image_count >= 5:
            break
        # y_true = y_true.cpu()
        # y_true = y_true.squeeze()
        # plt.imshow(y_true)
        # plt.savefig('orginal.png')
        x = x.to(torch.device(device))
        output = model(x)
        mask = np.digitize((output[0][0] * 47.83 * 12).detach().cpu().numpy(), np.array([1.5]), right=True)
        mask_float = np.float32(mask)
        image = torch.stack([x[0][0], x[0][0], x[0][0]], dim=2)
        image = image.cpu().numpy()
        targets = [SemanticSegmentationTarget(0, mask_float, device)]
        use_cuda = (device == 'cuda')
        cam_image = []
        for layer in target_layers:
            with GradCAM(model=model, target_layers=layer, use_cuda=use_cuda) as cam:
                grayscale_cam = cam(input_tensor=x, targets=targets)[0, :]

                cam_image.append(show_cam_on_image(image, grayscale_cam, use_rgb=True))

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # image_count += 1
        # axes[0].imshow(y_true[0].cpu().numpy())
        # axes[0].set_title('Ground Truth', {'fontsize': 16})
        # axes[1].imshow((output[0][0]).detach().cpu().numpy())
        # axes[1].set_title('Prediction', {'fontsize': 16})
        # axes[2].imshow(mask_float)
        # axes[2].set_title('Mask', {'fontsize': 16})

        # Encoder layers
        # axes[0].imshow(cam_image[0])
        # axes[0].set_title('CBAM 1', {'fontsize': 16})
        # axes[1].imshow(cam_image[1])
        # axes[1].set_title('CBAM 2', {'fontsize': 16})
        # axes[2].set_title('CBAM 3', {'fontsize': 16})
        # axes[2].imshow(cam_image[2])
        # axes[3].set_title('CBAM 4', {'fontsize': 16})
        # axes[3].imshow(cam_image[3])
        # axes[4].set_title('CBAM 5', {'fontsize': 16})
        # axes[4].imshow(cam_image[4])

        # Decoder layers
        axes[0].imshow(cam_image[2])
        axes[0].set_title('Up1', {'fontsize': 16})
        axes[1].imshow(cam_image[5])
        axes[1].set_title('Up2', {'fontsize': 16})
        axes[2].imshow(cam_image[8])
        axes[2].set_title('Up3', {'fontsize': 16})
        axes[3].imshow(cam_image[12])
        axes[3].set_title('Up4', {'fontsize': 16})
        axes[4].imshow(y_true)
        axes[4].set_title('Ground Truth', {'fontsize': 16})
        plt.savefig('heatmap.png')
        plt.show()


# Load the model
model = SmaAt_UNet(n_channels=12, n_classes=1)

# Load the state dictionary from the checkpoint file
checkpoint = torch.load(
    "/content/drive/MyDrive/UNetDS_Attention_rain_threshhold_50_epoch=52-val_loss=0.216322.ckpt")

# Sometimes the state dictionary is stored under the 'state_dict' key in the checkpoint
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

# Load the weights into the model
model.load_state_dict(checkpoint)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.eval()
model.to(torch.device(device))

data_file = "/content/drive/MyDrive/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"

dataset = dataset_precip.precipitation_maps_oversampled_h5(
    in_file=data_file,
    num_input_images=12,
    num_output_images=6, train=False)

test_dl = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

target_layers = [
    # Up1
    [model.up1],
    [model.up1.conv],
    [model.up1.conv.double_conv],

    # Up2
    [model.up2],
    [model.up2.conv],
    [model.up2.conv.double_conv],

    # Up3
    [model.up3],
    [model.up3.conv],
    [model.up3.conv.double_conv],
    # Up4
    [model.up4],  # The entire up4 module
    [model.up4.conv],  # The conv submodule in up4
    [model.up4.conv.double_conv],  # The double_conv submodule in up4
    # Optionally, you can target specific layers within the double_conv submodule
    [model.up4.conv.double_conv[0]],  # The first depthwise convolution in double_conv of up4
    [model.up4.conv.double_conv[4]],  # The second batch normalization in double_conv of up4
]

encoder_layers = [
    [model.cbam1],
    [model.cbam2],
    [model.cbam3],
    [model.cbam4],
    [model.cbam5],
]
# if one to run encoder: use encoder_layers
run_cam(model, target_layers, device, test_dl)

# input_img, target_img = dataset[0]  # Get the first sample in the dataset

# image_index = 1000  # choose the index of the image sequence to visualize
# input_images, _ = dataset[image_index]  # retrieve the input image sequence
#
# input_images_tensor = torch.from_numpy(input_images)
# input_images_tensor = input_images_tensor.unsqueeze(0)
#
# # Move input_images_tensor to GPU if available
# if torch.cuda.is_available():
#     input_images_tensor = input_images_tensor.cuda()
#
# #input_images = input_images.unsqueeze(0)  # add a batch dimension
#
# # Use GradCAM
# target_layer = moco_model.down4 # replace with the layer you want
# gradcam = GradCAM(model=moco_model,  target_layers=[target_layer], use_cuda=True)
# heatmap = gradcam(input_images_tensor)
# heatmap = heatmap - np.min(heatmap)
# heatmap = heatmap / np.max(heatmap)
#
# plt.imshow(input_images[-1].squeeze(), cmap='viridis')  # plot the last image in the sequence
# plt.savefig('heatmap.png')
# plt.imshow(heatmap.squeeze(), alpha=0.5, cmap='viridis', interpolation='bilinear')  # overlay the heatmap
# plt.savefig('heatmap2.png')  # save the plot as a PNG file
# plt.show()  # display the plot
