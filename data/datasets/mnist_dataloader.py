import logging
import os
import struct
import sys
import time
from random import random
import cv2
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from data.utils.split_train_val import split_train_val


class MNISTDataloader(Dataset):
    """
    Dataloader to load the traffic signs.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.positive_rate = self.config.positive_rate

        self.images = self.load_data()
        self.normalize()
        self.make_splitted_data()

    @property
    def data(self):
        return {'positive_pairs': self.positive_pairs, 'negative_pairs': self.negative_pairs,
                'tops': self.tops, 'bottoms': self.bottoms}

    def __getitem__(self, index):
        if self.split == 'train':
            if random() < self.positive_rate:
                pair = self.positive_pairs[index]
                top = self.tops[pair[0]]
                bottom = self.bottoms[pair[1]]
                label = 1.0
            else:
                pair_index = np.random.choice(
                    np.arange(len(self.negative_pairs)))  # there are sometimes more negatives than positives
                pair = self.negative_pairs[pair_index]
                top = self.tops[pair[0]]
                bottom = self.bottoms[pair[1]]
                label = 0.0
                pair = self.unshuffleing_pairs[index]

            return {'tops': top,
                    'bottoms': bottom,
                    'labels': np.array(label, dtype=np.float32),
                    'pairs': pair}

        elif self.split == 'val' or self.split == 'test':
            pair = self.negative_pairs[index]
            top = self.tops[pair[0]]
            bottom = self.bottoms[pair[1]]
            label = []
            pair = self.unshuffleing_pairs[index]

        else:
            raise ValueError(f'Wrong split: {self.split}')

        return {'tops': top, 'bottoms': bottom, 'pairs': pair, 'labels': label}

    def __len__(self):
        return len(self.images)

    def load_data(self):
        """
        Load data from the corresponding csv file.

        :return: data pandas dataframe
        """
        logging.info(f'Loading {self.split} data')

        if self.split == 'test':
            images_path = os.path.join(self.config.dataset_path, 't10k-images.idx3-ubyte')
            with open(images_path, 'rb') as imgpath:
                _ = struct.unpack(">IIII", imgpath.read(16))
                return np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28, 28)

        filaname = self.split + '_split.idx3-ubyte'
        images_path = os.path.join(self.config.dataset_path, filaname)
        if not os.path.exists(images_path):
            if self.split == 'train':
                split_train_val(self.config.train_val_split, self.config.dataset_path, 'train-images.idx3-ubyte')
            else:
                raise ValueError(f'Mode should be `train` for train-val split or `test` for testing on images.')

        assert os.path.exists(images_path), \
            f'The required data file path ({images_path}) does not exists'

        with open(images_path, 'rb') as imgpath:
            return np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28, 28)

    def collect_pairs(self, predicted_pairs):
        self.negative_pairs = np.empty([0, 2], dtype=np.int64)

        logging.info('Making new negative pairs from hard examples.')

        predicted_pairs = np.flip(predicted_pairs, -1)

        time.sleep(0.1)
        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(np.unique(predicted_pairs[:, 0]))), file=sys.stdout, bar_format=bar_format,
                  position=0, leave=True) as pbar:
            for i in np.unique(predicted_pairs[:, 0]):
                pairs_mask = predicted_pairs[:, 0] == i
                index = np.where(np.all(self.positive_pairs[i] == predicted_pairs[pairs_mask], axis=-1))
                negative = np.delete(predicted_pairs[pairs_mask], index, axis=0)
                self.negative_pairs = np.concatenate([self.negative_pairs, negative], axis=0)

                pbar.update(1)

        self.make_unshuffleing_list()

    def make_splitted_data(self):
        """
        Split the images
        """
        self.tops, self.bottoms = list(zip(*[(image[0, :14], image[0, 14:]) for image in self.images]))
        self.tops = np.expand_dims(np.array(self.tops, dtype=np.float32), 1)
        self.bottoms = np.expand_dims(np.array(self.bottoms, dtype=np.float32), 1)

        # if self.split == 'test':  # todo
        #     return

        # make pairs
        self.shuffle_indices()
        self.make_unshuffleing_list()

    def shuffle_indices(self):
        top_indices = list(range(self.__len__()))
        bottom_indices = list(range(self.__len__()))
        self.positive_pairs = np.array(list(zip(top_indices, bottom_indices)))

        # make negative pairs
        np.random.shuffle(top_indices)
        self.negative_pairs = np.array(list(zip(top_indices, bottom_indices)))

    def make_unshuffleing_list(self):
        unshuffleing_top = np.array([np.argwhere(self.negative_pairs[:, 0] == i)[0, 0]
                                      for i in range(len(self.negative_pairs))])
        self.unshuffleing_pairs = self.unshuffleing_pairs = np.stack([unshuffleing_top,
                                                                      np.arange(len(unshuffleing_top))], -1)

    def normalize(self):
        self.images = (np.expand_dims(np.array(self.images, dtype=np.float32), 1) - 33.3) / 76.8

    def unnormalize(self):
        self.images = (np.squeeze(self.images) * 76.8 + 33.3).astype(np.uint8)
        self.tops = (np.squeeze(self.tops) * 76.8 + 33.3).astype(np.uint8)
        self.bottoms = (np.squeeze(self.bottoms) * 76.8 + 33.3).astype(np.uint8)

    def save_test_images(self, predicted_tops):
        logging.info('Saving original, shuffled and reconstructed test images')

        test_images_dir = os.path.join(self.config.dataset_path, 'test_images')
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)

        self.unnormalize()
        for i, image in enumerate(self.images):
            filename = os.path.join(test_images_dir, f'{i}.jpg')
            cv2.imwrite(filename, image)

        for top_index, bottom_index in self.negative_pairs:
            image = np.concatenate([self.tops[top_index], self.bottoms[bottom_index]], 0)
            filename = os.path.join(test_images_dir, f'{bottom_index}_shuffled.jpg')
            cv2.imwrite(filename, image)

        for bottom_index, top_index in enumerate(predicted_tops):
            top_index = self.negative_pairs[:, 0][top_index]
            image = np.concatenate([self.tops[top_index], self.bottoms[bottom_index]], 0)
            filename = os.path.join(test_images_dir, f'{bottom_index}_reconstructed.jpg')
            cv2.imwrite(filename, image)
