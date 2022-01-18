import csv
import os
import struct

from sklearn.model_selection import train_test_split
import numpy as np




def split_train_val(split_rates, data_path, image_filename):
    """
    Split the data in the data dict according to the split rates
    """

    images_path = os.path.join(data_path, image_filename)
    assert os.path.exists(images_path), f'The data dir ({images_path}) does not exist.'

    # read the data and save the classnumber-classname relations csv
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        data = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28, 28)

    # make the splits and stratify to make sure that all the classes are represented in the val and test
    x_train, x_val, y_train, y_val = train_test_split(data, [None] * len(data), train_size=split_rates[0],
                                                      shuffle=True, random_state=0)
    x_train = x_train.reshape(-1)
    x_val = x_val.reshape(-1)

    # save the splits to csv files
    with open(os.path.join(data_path, 'train_split.idx3-ubyte'), 'wb') as f:
        x_train.tofile(f)
    with open(os.path.join(data_path, 'val_split.idx3-ubyte'), 'wb') as f:
        x_val.tofile(f)


