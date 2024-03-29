import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import pylab as plt


# This contains the whole training dataset (3 years)
class precipitation_maps_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images+num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images-(num_input_images+num_output_images)
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index:index+self.sequence_length], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.size_dataset


class precipitation_maps_oversampled_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.samples - 3822

class precipitation_maps_oversampled_h5_2years(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_2years, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        if index >= self.samples - 1911:
            raise IndexError("Index out of range of available data")

        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']

        # Shift the index by 3822
        shifted_index = index + 1911
        imgs = np.array(self.dataset[shifted_index], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.samples - 1911

class precipitation_maps_oversampled_h5_1years(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_1years, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        if index >= 1911:
            raise IndexError("Index out of range of available data")

        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024 ** 3)["train" if self.train else "test"][
                'images']

        # Shift the index by 3822
        shifted_index = index + 3822
        imgs = np.array(self.dataset[shifted_index], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return 1911

class precipitation_maps_oversampled_h5_pre(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5_pre, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms to each image separately
        transformed_images = []
        for img in imgs[:12]:
            if self.transform is not None:
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = Image.fromarray(img.astype(np.uint8), mode='L')
                #normalize first
                tensor1, tensor2 = self.transform(img)
                stacked_tensors = torch.stack([tensor1, tensor2], dim=0)
                transformed_images.append(stacked_tensors.numpy())
            # print(len(transformed_images))
            # print(type(transformed_images[0]))

        # stack the 12 channels together
        # input_img = torch.stack(transformed_images[:self.num_input], dim=0)
        input_img = (np.stack([x[0] for x in transformed_images], axis=0).squeeze(1),
             np.stack([x[1] for x in transformed_images], axis=0).squeeze(1))

        target_img = transformed_images[-1]


        return input_img, target_img

    def __len__(self):
        return self.samples




class precipitation_maps_classification_h5(Dataset):
    def __init__(self, in_file, num_input_images, img_to_predict, train=True, transform=None):
        super(precipitation_maps_classification_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape

        self.num_input = num_input_images
        self.img_to_predict = img_to_predict
        self.sequence_length = num_input_images + img_to_predict
        self.bins = np.array([0.0, 0.5, 1, 2, 5, 10, 30])

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images-(num_input_images + img_to_predict)
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index:index+self.sequence_length], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        # put the img in buckets
        target_img = imgs[-1]
        # target_img is normalized by dividing through the highest value of the training set. We reverse this.
        # Then target_img is in mm/5min. The bins have the unit mm/hour. Therefore we multiply the img by 12
        buckets = np.digitize(target_img*47.83*12, self.bins, right=True)

        return input_img, buckets

    def __len__(self):
        return self.size_dataset