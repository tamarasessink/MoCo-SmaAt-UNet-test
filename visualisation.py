import os
import argparse
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans

import moco.builder
from models.MoCo_SmaAt_Unet import MoCo_SmaAt_UNet
from utils import dataset_precip

import numpy as np
import torch
import random
import umap

if __name__ == "__main__":
    seed = 42  # You can choose any number as your seed
    # Set numpy random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

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
        MoCo_SmaAt_UNet(n_channels=12, n_classes=128),
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # model = MoCo_SmaAt_UNet(n_channels=12, n_classes=128)  # use this to see UMAP without pre-training

    # Use checkpoint of pre-trained network
    pretrained = '/content/drive/MyDrive/checkpoint_0199.pth (1).tar'
    pretrained = os.path.join(os.getcwd(), pretrained)
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # Load the pre-trained state dictionary and the target model's state dictionary
        state_dict_pre = checkpoint["state_dict"]
        state_dict_smaat_unet = model.state_dict()

        # Create a mapping dictionary that maps layer where the names correspond
        mapping_dict_pre = {k.replace("module.", ""): k for k in state_dict_pre.keys()}
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
                                               num_workers=0, pin_memory=True, drop_last=True)

    # Get the query encoder
    query_encoder = model.encoder_q

    model.eval()
    with torch.no_grad():
        features_list = []
        labels = []
        indexes = []
        image_list = []

        for i, (data, label) in enumerate(train_loader):
            # Extract features from the query encoder
            output, features = query_encoder.extract_features(data)
            features_list.append(features.cpu().numpy())
            image_list.append(data.cpu().numpy())  # Store the original images
            indexes.extend(range(i * train_loader.batch_size, (i + 1) * train_loader.batch_size))
            print(i)

    features_array = np.concatenate(features_list, axis=0)
    image_array = np.concatenate(image_list, axis=0)


    # Create the UMAP object and fit it to the data
    reducer = umap.UMAP()
    # Run UMAP on the features
    umap_results = reducer.fit_transform(features_array)

    # Fit KMeans with 6 clusters to your UMAP results
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(umap_results)

    # Plot the UMAP results
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.colorbar(label='Cluster Label')
    plt.title("UMAP Projections Clustered")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.savefig('UMAP.png')
    plt.show()

    clustered_images = {}  # Dictionary to hold the average image per cluster

    for i in range(6):  # Assuming you have 6 clusters
        indices = np.where(kmeans.labels_ == i)[0]
        cluster_images = image_array[indices]
        average_image = np.mean(cluster_images, axis=0)
        clustered_images[i] = average_image

    fig, axs = plt.subplots(1, 6, figsize=(20, 20))

    for i in range(6):  # Assuming you have 6 clusters
        average_image = clustered_images[i]
        if average_image.shape[0] == 1:
            average_image = np.squeeze(average_image, axis=0)
        axs[i].imshow(average_image)
        axs[i].set_title(f'Cluster {i + 1}')
        axs[i].axis('off')

    plt.show()

    # Create a DataFrame with the UMAP results and cluster labels
    df = pd.DataFrame({'UMAP-1': umap_results[:, 0], 'UMAP-2': umap_results[:, 1], 'Cluster': kmeans.labels_,
                       'index': range(len(umap_results))})

    # Create an interactive scatter plot
    fig = px.scatter(df, x='UMAP-1', y='UMAP-2', color='Cluster', color_continuous_scale='viridis',
                     hover_data=['index'])

    # Save the plot as an HTML file, to interactively explore the data points
    fig.write_html("umap_plot.html")
