from __future__ import print_function, division

import time
import os
import cv2
import torch.utils.data as data
import torch
import numpy as np

# use PIL Image to read image
def default_loader(path):
    # try:
    img = cv2.imread(path)
    # win = cv2.namedWindow('test win', flags=0)
    # cv2.imshow('test win', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, (227, 227))    # resize the image to 227x227x3
    return img
    # except:
    #     print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [image name/tab/label/tab/regression parameters], for example:/train/0001.jpg 0 1 1 1 1 1
class customData(data.Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            # self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_left_name = [line.strip().split('\t')[0] for line in lines]
            self.img_right_name = [line.strip().split('\t')[1] for line in lines]
            self.img_label_cls = [int(line.strip().split('\t')[2]) for line in lines]

        self.data_transforms = data_transforms
        self.loader = loader

    def __len__(self):
        return len(self.img_left_name)

    def __getitem__(self, item):
        img_left_name = self.img_left_name[item]
        img_right_name = self.img_right_name[item]
        label_cls = self.img_label_cls[item]

        img_left = self.loader(img_left_name)
        img_right = self.loader(img_right_name)

        if isinstance(img_left, np.ndarray):
            # handle numpy array
            img_left = torch.from_numpy(img_left.transpose((2, 0, 1)))
            # backward compatibility
            img_left = img_left.float().div(255)

        if isinstance(img_right, np.ndarray):
            # handle numpy array
            img_right = torch.from_numpy(img_right.transpose((2, 0, 1)))
            # backward compatibility
            img_right = img_right.float().div(255)

        label_cls = np.array(label_cls)
        if isinstance(np.array(label_cls), np.ndarray):
            # handle numpy array
            label_cls = torch.from_numpy(label_cls)
            # backward compatibility
            label_cls = label_cls.long()

        return img_left, img_right, label_cls
