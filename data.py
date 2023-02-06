import torch
from torch.utils.data import Dataset
import h5py
import tensorflow as tf
import numpy as np
from PIL import Image


# from VICTOR SCHMIDT — JUNE 15, 2021
class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive[f"trajectory_{i}"]
        data = torch.from_numpy(dataset[:])
        labels = dict(dataset.attrs)

        return {"data": data, "labels": labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


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


class Test(object):

    def __init__(self, data, transformm):
        self.data_arr = data  # define the data-array (load from file)
        self.transform = transformm

    def __getitem__(self, index):
        np_arr = self.data_arr[index, :]

        ## convert to PIL-image
        img = Image.fromarray((np_arr * 255).astype('uint8'))
        trans = self.transform(img)
        # apply the transformations and return tensors
        return trans


class createBatchDataset(np.ndarray):
    def __init__(self, data, end_last, batch_size, transform, new_data):
        self.data = data
        self.end_last = end_last
        self.batch_size = batch_size
        self.transform = transform
        self.new_data = new_data

    def __getitem__(self, num):
        for num in range(0, 32):
            # x, y = self.data[num]
            stack = self.data[num.astype(int)]
            i = 0
            while (i < 12):
                image = Image.fromarray(np.uint8(stack * 10000))
                q, k = self.transform(image)
                self.new_data[num][i] = q
                # aug_k[num][i] = q
                i = i + 1

        return self.new_data

