import os

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.SmaAt_Unet_pre_training import SmaAt_UNet_pre
from models.unet_precip_regression_lightning import UNetDS_Attention
from utils import dataset_precip

# Load the model
moco_model = SmaAt_UNet_pre(n_channels=12, n_classes=128) # replace with your architecture

# Load the checkpoint
pretrained = 'C:/Users/Tamara/Downloads/checkpoint_0199.pth (1).tar'
pretrained = os.path.join(os.getcwd(), pretrained)
if os.path.isfile(pretrained):
    print("=> loading checkpoint '{}'".format(pretrained))
    checkpoint = torch.load(pretrained, map_location="cpu")

    # Load the pre-trained state dictionary and the target model's state dictionary
    state_dict_pre = checkpoint["state_dict"]
    state_dict_smaat_unet = moco_model.state_dict()

    # Create a mapping dictionary that maps layer where the names correspond
    mapping_dict_pre = {k.replace("module.encoder_q.", ""): k for k in state_dict_pre.keys()}
    mapping_dict_smaat_unet = {k: k for k in state_dict_smaat_unet.keys()}

    # Create a new dictionary to store the updated keys
    updated_state_dict = {}

    # Iterate over the layer names and map the weights to layers with matching names
    for layer_name in mapping_dict_pre.keys():
        # Skip the fully connected layer of the pre-trained model, this is not in the target model
        if "fc" in layer_name:
            continue

        # Check if the layer names match and if the shapes match
        if layer_name in mapping_dict_smaat_unet and state_dict_pre[mapping_dict_pre[layer_name]].shape == \
                state_dict_smaat_unet[mapping_dict_smaat_unet[layer_name]].shape:
            updated_state_dict[mapping_dict_smaat_unet[layer_name]] = state_dict_pre[
                mapping_dict_pre[layer_name]]
            print(f"{mapping_dict_pre[layer_name]} mapped to {mapping_dict_smaat_unet[layer_name]}")
        else:
            print(f"Skipping {layer_name} due to mismatched shape or missing in the target model.")

    # Update the remaining keys in the target model that were not present in the pre-trained model
    for k in state_dict_smaat_unet.keys():
        if k not in updated_state_dict:
            updated_state_dict[k] = state_dict_smaat_unet[k]

    # Update the target model's state dictionary with the updated state dictionary, so that we don't miss any layer
    moco_model.load_state_dict(updated_state_dict)

moco_model = moco_model.eval()

# Assuming you're using a GPU
# moco_model = moco_model.cuda()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file = "/content/drive/MyDrive/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"

dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, train=False)

print(dataset[0])


image_index = 0  # choose the index of the image sequence to visualize
input_images, _ = dataset[image_index]  # retrieve the input image sequence

input_images = input_images.unsqueeze(0)  # add a batch dimension

# Use GradCAM
target_layer = moco_model.down4 # replace with the layer you want
gradcam = GradCAM(model=model, feature_module=target_layer, target_layer_names=['0'], use_cuda=True)
heatmap, result = gradcam(input_images)

plt.imshow(input_images.squeeze().permute(1, 2, 0)[dataset[image_index][11]])  # plot the last image in the sequence
plt.imshow(heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')  # overlay the heatmap
plt.show()  # display the plot

