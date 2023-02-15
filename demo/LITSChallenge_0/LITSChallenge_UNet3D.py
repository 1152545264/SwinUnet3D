import os

import torch
from torch import nn, functional as F, optim
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
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.config import KeysCollection
from torch.utils.data import random_split
from SwinUnet_3DV2 import swinUnet_t_3D
from monai.networks.nets import UNet, UNETR


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\LITSChallenge'
    window_center = min(30, 40)  # 针对CT而言的窗位和窗宽
    window_level = max(100, 200)
    HUMAX = 250
    HUMIN = -200
    # 数据集正向区域的shape中位数为[283,248,132]，但是FinaleShape设置为[256,256,128]加上半精度， batch=1,24G显存都不够
    FinalShape = [160, 160, 64]
    window_size = [5, 5, 2]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    train_ratio, val_ratio, test_ratio = [0.7, 0.2, 0.1]
    BatchSize = 1
    NumWorkers = 0
    n_classes = 2  # 分割总的类别数
    lr = 3e-4  # 学习率

    back_bone_name = 'UNETR'


class ReduceSegKinds(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ReduceSegKinds, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # print(d[key].shape)
            d[key][d[key] > 1] = 1
            # 输入是(X,Y,Z)
        return d


def cal_pos_region(d):
    # x:0 y:1 z:2
    z = np.any(d, axis=(0, 1))  # 寻找Z轴的非零区域
    z_tmp = np.where(z)
    z_start, z_end = z_tmp[0][[0, -1]]

    x = np.any(d, axis=(1, 2))
    x_tmp = np.where(x)
    x_start, x_end = x_tmp[0][[0, -1]]

    y = np.any(d, axis=(0, 2))
    y_tmp = np.where(y)
    y_start, y_end = y_tmp[0][[0, -1]]

    # d = d[x_start:x_end, y_start:y_end, z_start:z_end]

    return x_start, x_end, y_start, y_end, z_start, z_end


class CropPositiveRegion(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(CropPositiveRegion, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        x_start, x_end, y_start, y_end, z_start, z_end = cal_pos_region(d['seg'])
        for key in self.keys:
            d[key] = d[key][x_start:x_end, y_start:y_end, z_start:z_end]
        return d


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


class LitsDataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(LitsDataSet, self).__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.train_path = os.path.join(cfg.data_path, 'Training')
        self.test_path = os.path.join(cfg.data_path, 'Test')

        self.train_dict = []
        self.test_dict = []

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_process = None
        self.test_process = None

    def prepare_data(self):
        train_x, train_y, test_x = self.get_init()
        for x, y in zip(train_x, train_y):
            info = {'image': x, 'seg': y}
            self.train_dict.append(info)

        for x in test_x:
            info = {'image': x}
            self.test_dict.append(info)
        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.split_dataset()

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.test_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def get_preprocess(self):
        self.train_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            transforms.AddChanneld(keys=['image', 'seg']),

            # CropPositiveRegion(keys=['image', 'seg']),  # 裁剪出肝脏所在区域
            # transforms.AddChanneld(keys=['image', 'seg']),

            transforms.CropForegroundd(keys=['image', 'seg'], source_key='seg'),  # 裁剪出肝脏所在区域

            transforms.ResizeWithPadOrCropd(keys=['image', 'seg'], spatial_size=cfg.FinalShape),

            ReduceSegKinds(keys=['seg']),
            transforms.ScaleIntensityRanged(keys=['image'], a_max=self.cfg.HUMAX, a_min=self.cfg.HUMIN, b_max=1.0,
                                            b_min=0.0, clip=True),
            transforms.ToTensord(keys=['image', 'seg']),

        ])

        self.test_process = Compose([
            LoadImaged(keys=['image']),
            transforms.AddChanneld(keys=['image']),

            transforms.Resized(keys=['image'], spatial_size=self.cfg.FinalShape),

            # SelfResizeImaged(keys=['image', 'seg'], cfg=self.cfg),
            transforms.ScaleIntensityRanged(keys=['image'], a_max=self.cfg.HUMAX, a_min=self.cfg.HUMIN, b_max=1.0,
                                            b_min=0.0, clip=True),
            transforms.ToTensord(keys=['image', ]),
        ])

    def get_init(self):
        train_x = glob(os.path.join(self.train_path, 'volume-*.nii'))
        train_y = glob(os.path.join(self.train_path, 'segmentation-*.nii'))
        test_x = glob(os.path.join(self.test_path, 'test-volume-*.nii'))
        sorted(train_x)
        sorted(train_y)
        sorted(test_x)

        return train_x, train_y, test_x

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_set)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        test_num = int(num * cfg.test_ratio)
        if train_num + val_num + test_num != num:
            remain = num - train_num - test_num - val_num
            val_num += remain

        self.train_set, self.val_set, self.test_set = random_split(self.train_set, [train_num, val_num, test_num])


class LITSModel(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(LITSModel, self).__init__()
        self.cfg = cfg

        # self.net = UNet(spatial_dims=3, in_channels=1, out_channels=cfg.n_classes,
        #                 channels=[32, 64, 128, 256, 512], strides=(2, 2, 2, 2),
        #                 )

        self.net = UNETR(in_channels=1, out_channels=cfg.n_classes, img_size=cfg.FinalShape)

        self.loss_func = monai.losses.DiceCELoss(to_onehot_y=True)
        self.metrics = monai.metrics.DiceMetric()

    def forward(self, x):
        x = self.net(x)
        return x

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}

    def training_step(self, batch, batch_idx):
        dice_loss, dice = self.shared_step(batch)
        self.log('train_loss', dice_loss, prog_bar=False)
        self.log('train_dice', dice, prog_bar=False)
        return {'loss': dice_loss, 'train_dice': dice}

    def validation_step(self, batch, batch_idx):
        dice_loss, dice = self.shared_step(batch)
        self.log('valid_loss', dice_loss, prog_bar=False)
        self.log('valid_dice', dice, prog_bar=False)
        return {'valid_loss': dice_loss, 'valid_dice': dice}
        # return dice_loss, dice

    def test_step(self, batch, batch_idx):
        dice_loss, dice = self.shared_step(batch)
        self.log('test_loss', dice_loss, prog_bar=False)
        self.log('test_dice', dice, prog_bar=False)
        # return dice_loss, dice
        return {'test_loss': dice_loss, 'test_dice': dice}

    def training_epoch_end(self, outputs):
        train_losses, train_dices = self.shared_epoch_end(outputs, 'loss', 'train_dice')
        if len(train_losses) > 0:
            train_mean_loss = np.mean(train_losses)
            train_mean_dice = np.mean(train_dices)
            self.log('train_mean_loss', train_mean_loss, prog_bar=True)
            self.log('train_mean_dice', train_mean_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        valid_losses, valid_dices = self.shared_epoch_end(outputs, 'valid_loss', 'valid_dice')
        if len(valid_losses) > 0:
            valid_mean_loss = np.mean(valid_losses)
            valid_mean_dice = np.mean(valid_dices)
            self.log('valid_mean_loss', valid_mean_loss, prog_bar=True)
            self.log('valid_mean_dice', valid_mean_dice, prog_bar=True)

    def test_epoch_end(self, outputs):
        test_losses, test_dices = self.shared_epoch_end(outputs, 'test_loss', 'test_dice')
        if len(test_losses) > 0:
            test_mean_loss = np.mean(test_losses)
            test_mean_dice = np.mean(test_dices)
            self.log('test_mean_loss', test_mean_loss, prog_bar=True)
            self.log('test_mean_dice', test_mean_dice, prog_bar=True)

    @staticmethod
    def shared_epoch_end(outputs, loss_key, dice_key):
        losses, dices = [], []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key].item()
            dice = output[dice_key].item()

            losses.append(loss)
            dices.append(dice)
        losses = np.array(losses)
        dices = np.array(dices)
        return losses, dices

    def shared_step(self, batch):
        x = batch['image']
        y = batch['seg']
        y_hat = self.net(x)
        dice_loss = self.loss_func(y_hat, y)

        # y_hat = torch.argmax(y_hat, dim=1)
        # y_hat = torch.unsqueeze(y_hat, dim=1)

        # y_hat = torch.softmax(y_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        y_hat = torch.unsqueeze(y_hat, dim=1)
        dice = self.metrics(y_hat, y)
        return dice_loss, dice


pl.seed_everything(42)

data = LitsDataSet()
model = LITSModel()

early_stop = EarlyStopping(
    monitor='valid_mean_loss',
    patience=10,
)

cfg = Config()
check_point = ModelCheckpoint(dirpath=f'./trained_models/{cfg.back_bone_name}',
                              save_last=False,
                              save_top_k=4, monitor='valid_mean_loss', verbose=True,
                              filename='{epoch}-{valid_loss:.2f}-{valid_mean_dice:.2f}')
trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=200,
    min_epochs=30,
    gpus=1,
    # auto_select_gpus=True, # 这个参数针对混合精度训练时，不能使用

    # auto_lr_find=True,
    auto_scale_batch_size=True,
    logger=TensorBoardLogger(save_dir=f'./logs', name=f'{cfg.back_bone_name}'),
    callbacks=[early_stop, check_point],
    precision=16,
    accumulate_grad_batches=4,
    num_sanity_val_steps=0,
    log_every_n_steps=50,
    auto_lr_find=True
)

NeedTrain = True

if NeedTrain:
    trainer.fit(model, data)
