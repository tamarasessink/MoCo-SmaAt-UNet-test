import os

import torch

from models import unet_precip_regression_lightning as unet_regr


def get_model_class(model_file):
    # This is for some nice plotting
    if "UNet_Attention" in model_file:
        model_name = "UNet Attention"
        model = unet_regr.UNet_Attention
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "BackbonedUNet" in model_file:
        model_name = "ResNet with UNet"
        model = unet_regr.BackbonedUNet
    elif "UNetDS_Attention_1kpl" in model_file:
        model_name = "UNetDS Attention with 1kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
        pretrained = '/content/drive/MyDrive/models/checkpoint_0151_0.001.pth.tar'
        pretrained = os.path.join(os.getcwd(), pretrained)
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # Load the pre-trained state dictionary and the target model's state dictionary
            state_dict_pre = checkpoint["state_dict"]
            state_dict_smaat_unet = model.state_dict()

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
            model.load_state_dict(updated_state_dict)
    elif "UNetDS" in model_file:
        model_name = "UNetDS"
        model = unet_regr.UNetDS
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError(f"Model not found")
    return model, model_name