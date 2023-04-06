import os

import torch
from torch.utils.data import Dataset
import h5py
import tensorflow as tf
import numpy as np
from PIL import Image
import pylab as plt


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, lags):
        self.data = data
        self.b_size = batch_size
        self.lags = lags
        self.time_steps = data.shape[0]

    # Calculates the number of batches: samples/batch_size
    def __len__(self):
        # Calculating the number of batches
        return int(self.time_steps / self.b_size)

    # Obtains one batch of data
    def __getitem__(self, idx):
        x = self.data[idx * self.b_size:(idx + 1) * self.b_size, 0:self.lags, :, :]
        y = self.data[idx * self.b_size:(idx + 1) * self.b_size, -1, :, :]

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        y = np.expand_dims(y, axis=1)

        return x, y


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip



class createDataset:
    def __init__(self):
        self.filepath = "/content/drive/MyDrive/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
        self.f = h5py.File(self.filepath, "r")
        self.traindir = self.f['/train/images']
        self.valdir = self.f['/test/images']
        self.trainlen = self.traindir.shape[0]
        self.vallen = self.valdir.shape[0]
        self.training_generator_train = DataGenerator(self.traindir, 1, 12)
        self.training_generator_val = DataGenerator(self.valdir, 1, 12)

    def process_images_train(self):
        for num in range(0, self.trainlen):
            x, y = self.training_generator_train[num]
            stack = x.squeeze()
            i = 0
            image_folder = "/image" + str(num)
            directory = "images" + image_folder
            if not os.path.exists(directory):
                os.makedirs(directory)

            while (i < 12):
                min_val = np.min(stack[i])
                max_val = np.max(stack[i])
                # normalizing the data
                normalized = (stack[i] - min_val) / (max_val - min_val) * 255.0
                image = Image.fromarray(np.uint8(normalized))
                image_num = "/image" + i * 'I' + ".jpeg"
                if not os.path.exists(image_num):
                    plt.imsave(directory + image_num, image)
                i = i + 1

    def process_images_test(self):
        for num in range(0, self.vallen):
            x, y = self.training_generator_val[num]
            stack = x.squeeze()
            i = 0
            image_folder = "/image" + str(num)
            directory = "images_val" + image_folder
            if not os.path.exists(directory):
                os.makedirs(directory)

            while (i < 12):
                min_val = np.min(stack[i])
                max_val = np.max(stack[i])
                # normalizing the data
                normalized = (stack[i] - min_val) / (max_val - min_val) * 255.0
                image = Image.fromarray(np.uint8(normalized))
                image_num = "/image" + i * 'I' + ".jpeg"
                if not os.path.exists(image_num):
                    plt.imsave(directory + image_num, image)
                i = i + 1




