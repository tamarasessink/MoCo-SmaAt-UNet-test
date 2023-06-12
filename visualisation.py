import os

import numpy as np
import umap.umap_ as umap
import argparse
import pytorch_lightning as pl
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from torch import device
import matplotlib.cm as cm
import plotly.express as px
import pandas as pd

import moco.loader
import moco.builder
from models import unet_precip_regression_lightning as unet_regr
from models.SmaAt_UNet import SmaAt_UNet
from models.SmaAt_Unet_pre_training import SmaAt_UNet_pre
from utils import dataset_precip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_checkpoint', type=str,
                        default='C:/Users/Tamara/Downloads/checkpoint_0199.pth (1).tar')
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.10, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')

    args = parser.parse_args()

    model = moco.builder.MoCo(
        SmaAt_UNet_pre(n_channels=12, n_classes=128),
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # model = SmaAt_UNet_pre(n_channels=12, n_classes=128)  # use the appropriate model class here

    pretrained = '/content/drive/MyDrive/checkpoint_0199.pth (1).tar'
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

    dataset = '/content/drive/MyDrive/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5'

    train_dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file=dataset, num_input_images=12,
        num_output_images=6, train=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False,

                                               num_workers=2, pin_memory=True, drop_last=True)

    # Get the query encoder
    query_encoder = model.encoder_q

    # Assume dataloader is an instance of torch.utils.data.DataLoader that loads your images
    model.eval()
    with torch.no_grad():
        features_list = []
        labels = []
        indexes = []

        for i, (data, label) in enumerate(train_loader):
            # Extract features from the query encoder
            output, features = query_encoder.extract_features(data)
            features_list.append(features.cpu().numpy())
            indexes.extend(range(i * train_loader.batch_size, (i + 1) * train_loader.batch_size))

            print(i)

    features_array = np.concatenate(features_list, axis=0)
    # labels = np.concatenate(labels, axis=0)

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(features)
    #
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # clusters = kmeans.fit_predict(tsne_results)

    # Create the UMAP object and fit it to your data
    reducer = umap.UMAP()
    # Run UMAP on the features
    umap_results = reducer.fit_transform(features_array)
    # Create a figure
    plt.figure(figsize=(10, 10))

    # Plot the features in red
    plt.scatter(umap_results[:, 0], umap_results[:, 1], alpha=0.5, label='Features')

    # index_cluster0 = np.where(clusters == 0)[0][4]
    # index_cluster1 = np.where(clusters == 1)[0][4]
    #
    # # Fetch the corresponding images using the indices from the clusters and the original_data array
    # image_cluster0 = original_data[index_cluster0]
    # image_cluster1 = original_data[index_cluster1]
    #
    # # Display the images as before
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_cluster0[0])
    # plt.title('First image from cluster 0')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_cluster1[0])
    # plt.title('First image from cluster 1')
    # Create a DataFrame with the UMAP results
    df = pd.DataFrame({'UMAP-1': umap_results[:, 0], 'UMAP-2': umap_results[:, 1], 'index': range(len(umap_results))})

    # Create an interactive scatter plot
    fig = px.scatter(df, x='UMAP-1', y='UMAP-2', hover_data=['index'])

    # Save the plot as an HTML file
    fig.write_html("umap_plot.html")
    plt.savefig('UMAP.png')
    plt.show()

    # index_a = # index of the point in cluster A
    # index_b = # index of the point in cluster B

    # # Load the actual images or segmentation maps using the indexes
    # image_a, _ = train_dataset[indexes[index_a]]
    # image_b, _ = train_dataset[indexes[index_b]]

    # # Visualization
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(image_a[0], cmap='gray')
    # plt.title('Image from cluster A')

    # plt.subplot(1, 2, 2)
    # plt.imshow(image_b[0], cmap='gray')
    # plt.title('Image from cluster B')

    # plt.show()





