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
from SwinUnet_3DV2 import swinUnet_t_3D
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch, list_data_collate
from monai.networks.nets import UNETR, UNet, VNet
from timm.models.layers import trunc_normal_
from monai.data import NiftiSaver
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandCropByLabelClassesd
)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\LITSChallenge'
    Seed = 42
    # 肝脾CT检查应适当变窄窗宽以便更好发现病灶，窗宽为100 Hu~200 Hu,窗位为30 Hu~45 Hu,
    HuMax = 180
    HuMin = -70
    LowPercent = 0.5
    HighPercent = 99.5

    in_channels = 1
    num_samples = 4
    FinalShape = [96, 96, 64]
    window_size = [it // 32 for it in FinalShape]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    slid_window_overlap = 0.5

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0
    n_classes = 3  # 分割总的类别数:背景+肝脏+肿瘤
    lr = 3e-4  # 学习率
    PixDim = (2.0, 2.0, 1.0)

    # back_bone_name = 'SwinUnet3D'
    back_bone_name = 'Unet3D'


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(f'\n{key}的形状为：{d[key].shape}')
            # 输入是(X,Y,Z)
        return d


class LitsDataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(LitsDataSet, self).__init__()
        self.cfg = cfg
        self.train_path = cfg.data_path

        self.train_dict = []
        self.val_dict = []
        self.test_dict = []

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_process = None
        self.val_process = None

    def prepare_data(self):
        # train_x, train_y, test_x = self.get_init()
        train_x, train_y = self.get_init()
        for x, y in zip(train_x, train_y):
            info = {'image': x, 'label': y}
            self.train_dict.append(info)

        # for x in test_x:
        #     info = {'image': x}
        #     self.test_dict.append(info)

        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.split_dataset()
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.test_set = Dataset(self.test_dict, transform=self.val_process)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers, shuffle=True,
                          collate_fn=list_data_collate)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def test_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.test_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def get_preprocess(self):
        self.train_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            Spacingd(keys=['image', 'label'], pixdim=self.cfg.PixDim,
                     mode=('bilinear', 'nearest')),

            # ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.LowPercent,
            #                                 upper=cfg.HighPercent, relative=True,
            #                                 b_min=0.0, b_max=1.0, clip=True),

            ScaleIntensityRanged(keys='image', a_min=cfg.HuMin, a_max=cfg.HuMax,
                                 b_min=0.0, b_max=1.0, clip=True),

            # CropForegroundd(keys=["image", "label"], source_key='image'),
            # RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
            #                        spatial_size=cfg.FinalShape, pos=1, neg=1,
            #                        num_samples=cfg.num_samples, image_key='image', ),

            RandCropByLabelClassesd(keys=['image', 'label'], label_key='label',
                                    spatial_size=cfg.FinalShape, ratios=(1, 1, 1),
                                    num_classes=cfg.n_classes, num_samples=cfg.num_samples,
                                    image_key='image'
                                    ),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3),

            EnsureTyped(keys=['image', 'label']),

        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            Spacingd(keys=['image', 'label'], pixdim=self.cfg.PixDim,
                     mode=('bilinear', 'nearest')),

            ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.LowPercent,
                                            upper=cfg.HighPercent, relative=True,
                                            b_min=0.0, b_max=1.0, clip=True),
            # 验证集没有做空间尺寸的变换，是因为在模型的valid_step中，我们使用了滑动窗口进行推理,即函数sliding_window_inference

            EnsureTyped(keys=['image', 'label']),
        ])

    def get_init(self):
        train_x = sorted(glob(os.path.join(self.train_path, 'volume-*.nii')))
        train_y = sorted(glob(os.path.join(self.train_path, 'segmentation-*.nii')))

        # train_x.sort()
        # train_y.sort()

        return train_x, train_y

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_dict)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        test_num = int(num * cfg.test_ratio)
        if train_num + val_num + test_num != num:
            remain = num - train_num - test_num - val_num
            val_num += remain

        self.train_dict, self.val_dict, self.test_dict = random_split(self.train_dict,
                                                                      [train_num, val_num, test_num],
                                                                      generator=torch.Generator().manual_seed(cfg.Seed))


class LITSModel(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(LITSModel, self).__init__()
        self.cfg = cfg

        if cfg.back_bone_name == 'SwinUnet3D':
            self.net = swinUnet_t_3D(window_size=cfg.window_size, num_classes=cfg.n_classes, in_channel=cfg.in_channels)
        else:
            from monai.networks.nets import UNETR, UNet
            self.net = UNETR(in_channels=cfg.in_channels, out_channels=cfg.n_classes, img_size=cfg.FinalShape)

        self.loss_func = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=cfg.n_classes)
        ])
        self.post_label = Compose([
            EnsureType(),
            AsDiscrete(to_onehot=cfg.n_classes)
        ])

        self.metrics = DiceMetric(include_background=False, reduction="mean_batch",
                                  get_not_nans=False)
        # self.metrics = dice_score

    def forward(self, x):
        # x = self.net(x)
        y_hat = sliding_window_inference(x, roi_size=cfg.FinalShape, sw_batch_size=cfg.BatchSize, predictor=self.net,
                                         overlap=cfg.slid_window_overlap)
        return y_hat

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=1, )
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_loss'}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        assert x.shape[2:] == y.shape[2:]
        y_hat = self.net(x)

        dice_loss, dice = self.shared_step(y_hat=y_hat, y=y)

        l_dice, t_dice = dice[0], dice[1]
        self.log('train_loss', dice_loss, prog_bar=True)
        self.log('train_liver_dice', l_dice, prog_bar=True)
        self.log('train_tumor_dice', t_dice, prog_bar=True)
        return {'loss': dice_loss}

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        assert x.shape[2:] == y.shape[2:]
        y_hat = self.forward(x)
        dice_loss, dice = self.shared_step(y_hat=y_hat, y=y)
        l_dice, t_dice = dice[0], dice[1]
        self.log('valid_loss', dice_loss, prog_bar=False)
        self.log('valid_liver_dice', l_dice, prog_bar=False)
        self.log('valid_tumor_dice', t_dice, prog_bar=True)
        return {'loss': dice_loss}

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label']
        y_hat = self.forward(x)
        dice_loss, dice = self.shared_step(y_hat, y)
        self.log('valid_loss', dice_loss, prog_bar=False)
        self.log('valid_dice', dice, prog_bar=False)
        return {'loss': dice_loss}

    def training_epoch_end(self, outputs):
        mean_loss, mean_dice = self.shared_epoch_end(outputs, 'loss')
        l_mean_dice, t_mean_dice = mean_dice[0], mean_dice[1]

        self.log('train_mean_loss', mean_loss, prog_bar=True)
        self.log('train_mean_liver_dice', l_mean_dice, prog_bar=True)
        self.log('train_mean_tumor_dice', t_mean_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        mean_loss, mean_dice = self.shared_epoch_end(outputs, 'loss')
        l_mean_dice, t_mean_dice = mean_dice[0], mean_dice[1]

        self.log('valid_mean_loss', mean_loss, prog_bar=True)
        self.log('valid_mean_liver_dice', l_mean_dice, prog_bar=True)
        self.log('valid_mean_tumor_dice', t_mean_dice, prog_bar=True)

    def test_epoch_end(self, outputs):
        mean_loss, mean_dice = self.shared_epoch_end(outputs, 'loss')
        l_mean_dice, t_mean_dice = mean_dice[0], mean_dice[1]

        self.log('test_mean_loss', mean_loss, prog_bar=True)
        self.log('test_mean_liver_dice', l_mean_dice, prog_bar=True)
        self.log('test_mean_tumor_dice', t_mean_dice, prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key):
        losses = []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key].item()
            losses.append(loss)
        losses = np.array(losses)
        mean_loss = np.mean(losses)

        mean_dice = self.metrics.aggregate()
        self.metrics.reset()

        mean_dice = mean_dice.detach().cpu().numpy()
        return mean_loss, mean_dice

    def shared_step(self, y_hat, y):
        dice_loss = self.loss_func(y_hat, y)

        from monai.data import decollate_batch
        y_hat = [self.post_pred(i) for i in decollate_batch(y_hat)]
        y = [self.post_label(i) for i in decollate_batch(y)]
        dice = self.metrics(y_hat, y)
        dice = torch.mean(dice, dim=0)
        dice_loss = torch.nan_to_num(dice_loss)  # 避免某一次loss为nan造成后续训练全部崩掉
        dice = torch.nan_to_num(dice)
        # dice = torch.mean(dice)

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
                              filename='{epoch}-{valid_loss:.2f}-{valid_mean_tumor_dice:.2f}')
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
