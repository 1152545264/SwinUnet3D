import os
import random

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
from SwinUnet_3D import swinUnet_t_3D
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from torchmetrics.functional import dice_score
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\MSD\lung'

    FinalShape = [160, 160, 160]  # 尺寸中位数为512,512,252
    window_size = [5, 5, 5]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 1

    OriginPixDim = (0.7, 0.7, 1.0)
    ResamplePixDim = (1.5, 1.5, 2.0)

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0

    lower = 5
    upper = 95

    n_classes = 2

    lr = 3e-4  # 学习率

    back_bone_name = 'SwinUnet_0'
    # back_bone_name = 'Unet3D_0'
    # back_bone_name = 'UnetR'

    # 滑动窗口推理时使用
    roi_size = FinalShape
    slid_window_overlap = 0.5


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
        self.train_path = os.path.join(cfg.data_path, 'imagesTr')
        self.label_tr_path = os.path.join(cfg.data_path, 'labelsTr')
        self.test_path = os.path.join(cfg.data_path, 'imagesTs')

        self.train_dict = []
        self.val_dict = []
        self.test_dict = []

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_process = None
        self.val_process = None

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
        self.split_dataset()
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.test_set = Dataset(self.test_dict, transform=self.val_process)

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
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            transforms.EnsureChannelFirstd(keys=['image', 'seg']),

            transforms.Spacingd(keys=['image', 'seg'], pixdim=cfg.ResamplePixDim,
                                mode=['bilinear', 'nearest']),
            transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),

            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.lower, upper=cfg.upper, b_min=0.0,
                                                       b_max=1.0, clip=True),
            transforms.CropForegroundd(keys=['image', 'seg'], source_key='image'),
            transforms.RandSpatialCropd(keys=['image', 'seg'], roi_size=cfg.FinalShape, random_size=False),
            transforms.SpatialPadd(keys=['image', 'seg'], spatial_size=cfg.FinalShape),

            transforms.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),

            transforms.ToTensord(keys=['image', 'seg']),

        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            transforms.EnsureChannelFirstd(keys=['image', 'seg']),

            transforms.Spacingd(keys=['image', 'seg'], pixdim=cfg.ResamplePixDim, mode=['bilinear', 'nearest']),
            transforms.Orientationd(keys=['image', 'seg'], axcodes='RAS'),

            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.lower, upper=cfg.upper, b_min=0.0,
                                                       b_max=1.0, clip=True),
            transforms.ToTensord(keys=['image', 'seg']),
        ])

    def get_init(self):
        train_x = sorted(glob(os.path.join(self.train_path, '*.nii.gz')))
        train_y = sorted(glob(os.path.join(self.label_tr_path, '*.nii.gz')))
        test_x = sorted(glob(os.path.join(self.test_path, '*.nii.gz')))

        # train_x.sort()
        # train_y.sort()
        # test_x.sort()
        return train_x, train_y, test_x

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_dict)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        test_num = int(num * cfg.test_ratio)
        if train_num + val_num + test_num != num:
            remain = num - train_num - test_num - val_num
            val_num += remain

        self.train_dict, self.val_dict, self.test_dict = random_split(self.train_dict, [train_num, val_num, test_num])


class Lung(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(Lung, self).__init__()
        self.cfg = cfg
        if cfg.back_bone_name == 'SwinUnet_0':
            self.net = swinUnet_t_3D(window_size=cfg.window_size, num_classes=cfg.n_classes, in_channel=cfg.in_channels,
                                     )
        else:
            from monai.networks.nets import UNETR, UNet
            # self.net = UNETR(in_channels=cfg.in_channels, out_channels=cfg.n_classes, img_size=cfg.FinalShape)
            self.net = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256, 512),
                            strides=(2, 2, 2, 2))

        self.loss_func = GeneralizedDiceLoss(to_onehot_y=True, softmax=True, include_background=False)

        self.post_pred = Compose([
            transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=True, num_classes=cfg.n_classes)
        ])
        self.post_label = Compose([transforms.EnsureType(),
                                   transforms.AsDiscrete(to_onehot=True, num_classes=cfg.n_classes)])

        # 这个类有bug, 当y_pred和y均为全零矩阵时，此处计算出来的dice系数为nan
        self.metrics = DiceMetric(include_background=False, reduction="mean")
        # self.metrics = dice_score

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['seg']
        assert x.shape == y.shape
        y_hat = self.net(x)
        dice_loss, dice = self.shared_step(y_hat=y_hat, y=y)
        self.log('train_loss', dice_loss, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return {'loss': dice_loss, 'train_dice': dice}

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['seg']

        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=cfg.BatchSize, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        dice_loss, dice = self.shared_step(y_hat=y_hat, y=y)
        self.log('valid_loss', dice_loss, prog_bar=True)
        self.log('valid_dice', dice, prog_bar=True)
        return {'valid_loss': dice_loss, 'valid_dice': dice}

    def test_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['seg']
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=1, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        dice_loss, dice = self.shared_step(y_hat, y)
        self.log('test_loss', dice_loss, prog_bar=True)
        self.log('test_dice', dice, prog_bar=True)
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
        self.metrics.reset()
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

    def shared_step(self, y_hat, y):
        dice_loss = self.loss_func(y_hat, y)

        y_hat = [self.post_pred(i) for i in decollate_batch(y_hat)]
        y = [self.post_label(i) for i in decollate_batch(y)]
        dice = self.metrics(y_hat, y)
        dice = torch.mean(dice, dim=0)
        dice = torch.nan_to_num(dice)
        return dice_loss, dice


data = LitsDataSet()
model = Lung()

early_stop = EarlyStopping(
    monitor='valid_mean_loss',
    patience=10,
)

cfg = Config()
check_point = ModelCheckpoint(dirpath=f'./trained_models/{cfg.back_bone_name}',
                              save_last=False,
                              save_top_k=2, monitor='valid_mean_loss', verbose=True,
                              filename='{epoch}-{valid_loss:.2f}-{valid_mean_dice:.2f}')
trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=1000,
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
    log_every_n_steps=400,
    auto_lr_find=True
)
trainer.fit(model, data)
