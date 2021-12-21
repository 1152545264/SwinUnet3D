import os

import torch
from torch import nn, functional as F
import monai
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import Compose
from monai.transforms import LoadImaged, LoadImage
from monai.data import Dataset, SmartCacheDataset
from torch.utils.data import DataLoader, random_split
from glob import glob
import numpy as np
from monai.config import KeysCollection

data_path = r'D:\Caiyimin\Dataset\MSD\BrainTumor'
window_center = min(30, 40)
window_level = max(100, 200)
HUMAX = 250
HUMIN = -200


class CropPositiveRegion(transforms.Transform):

    def __init__(self, **kwargs):
        super(CropPositiveRegion, self).__init__(**kwargs)

    def __call__(self, data):
        d = data
        x_s, y_s, z_s = d.shape  # x:0 y:1 z:2
        z = np.any(d, axis=(0, 1))  # 寻找Z轴的非零区域
        z_tmp = np.where(z)
        z_start, z_end = z_tmp[0][[0, -1]]

        x = np.any(d, axis=(1, 2))
        x_tmp = np.where(x)
        x_start, x_end = x_tmp[0][[0, -1]]

        y = np.any(d, axis=(0, 2))
        y_tmp = np.where(y)
        y_start, y_end = y_tmp[0][[0, -1]]

        d = d[x_start:x_end, y_start:y_end, z_start:z_end]

        return d


def analysis_data():
    shape_x, shape_y, shape_z = [], [], []
    pixel_max, pixel_min = [], []
    train_path = os.path.join(data_path, 'imagesTr')
    train_files = glob(os.path.join(train_path, '*.nii.gz'))
    for file in train_files:
        img_tmp = LoadImage()(file)
        img = img_tmp[0]
        '''
        loader = LoadImage()
        img = loader(file)
        '''

        x, y, z, c = img.shape
        pmax, pmin = np.max(img), np.min(img)
        shape_x.append(x)
        shape_y.append(y)
        shape_z.append(z)
        pixel_max.append(pmax)
        pixel_min.append(pmin)

    shape_x, shape_y, shape_z = np.array(shape_x), np.array(shape_y), np.array(shape_z)
    pixel_max, pixel_min = np.array(pixel_max), np.array(pixel_min)

    shape_z_median = np.median(shape_z)
    shape_x_median = np.median(shape_x)
    shape_y_median = np.median(shape_y)

    sorted(shape_z), sorted(shape_x), sorted(shape_y)

    with open(os.path.join(train_path, 'dataset_info.txt'), 'w+', encoding='utf-8') as f:
        f.write(f'Z轴中位数为：{shape_z_median}\n')
        f.write(f'X轴中位数为：{shape_x_median}\n')
        f.write(f'Y轴中位数为：{shape_y_median}\n')
        f.write(f'Z轴所有shape为：{shape_z}\n')
        f.write(f'像素最大值为：{pixel_max}\n')
        f.write(f'像素最小值为：{pixel_min}\n')


def analysis_label():
    train_path = os.path.join(data_path, 'labelsTr')
    train_files = glob(os.path.join(train_path, '*.nii.gz'))
    for file in train_files:
        label_tmp = LoadImage()(file)
        label = label_tmp[0]
        print(np.max(label))
        print(np.min(label))


def arr2dict(shape):
    m = {}
    for x in shape:
        try:
            m[x] += 1
        except Exception as e:
            m.update({x: 1})
    return m


def analysis_data_pos_region():
    shape_x, shape_y, shape_z = [], [], []
    train_path = os.path.join(data_path, 'labelsTr')
    train_files = glob(os.path.join(train_path, '*.nii.gz'))
    for file in train_files:
        img_tmp = LoadImage()(file)
        img = img_tmp[0]
        img = CropPositiveRegion()(img)
        x, y, z = img.shape

        shape_x.append(x)
        shape_y.append(y)
        shape_z.append(z)

    shape_x, shape_y, shape_z = np.array(shape_x), np.array(shape_y), np.array(shape_z)
    x_median, y_median, z_median = np.median(shape_x), np.median(shape_y), np.median(shape_z)
    with open(os.path.join(train_path, 'dataset_info'), 'a+', encoding='utf-8') as f:
        for i in range(3):
            f.write('\n')
        f.write(f'Z轴正向区域范围中位数为：{z_median}\n')
        f.write(f'X轴正向区域范围中位数为：{x_median}\n')
        f.write(f'Y轴正向区域范围中位数为：{y_median}\n')
        f.write(f'Z轴正向区域范围为：{shape_z}\n')
        f.write(f'X轴正向区域范围为：{shape_x}\n')
        f.write(f'Y轴正向区域范围为：{shape_y}\n')


def testCropPositive(seg_path: str = os.path.join(data_path, r'Training\segmentation-0.nii')):
    seg_img = LoadImage()(seg_path)[0]
    seg_img = CropPositiveRegion()(seg_img)
    print(seg_img.shape)


# testCropPositive()
# analysis_data()
analysis_label()
# analysis_data_pos_region()
