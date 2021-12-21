# -*-coding:utf-8-*-
import os
import random

import torch
from torch import nn, functional as F, optim
import monai
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import Compose, ToTensord
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
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric
from torchmetrics.functional import dice_score
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch
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
    ConvertToMultiChannelBasedOnBratsClassesd,
    SpatialPadd,
    ScaleIntensityRangePercentilesd
)

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    data_path = r'D:\Caiyimin\Dataset\Brats2021'
    # 脑组织窗宽设定为80Hu~100Hu, 窗位为30Hu~40Hu,
    PadShape = [240, 240, 160]
    FinalShape = [224, 224, 160]
    window_size = [7, 7, 5]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 4

    l_percent = 0.5
    u_percent = 99.5

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 0
    '''
    标签中绿色为浮肿区域(ED,peritumoral edema) (标签2)、黄色为增强肿瘤区域(ET,enhancing tumor)(标签4)、
    红色为坏疽(NET,non-enhancing tumor)(标签1)、背景(标签0)
    WT = ED + ET + NET
    TC = ET+NET
    ET
    '''
    n_classes = 3  # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、ET（肿瘤增强)和WT（整个肿瘤）。

    lr = 3e-4  # 学习率
    back_bone_name = 'SwinUnet'

    # 滑动窗口推理时使用
    roi_size = FinalShape
    overlap = 0.0


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


class Brats2021(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(Brats2021, self).__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.train_path = os.path.join(cfg.data_path, 'Brats2021Train')
        # self.label_tr_path = os.path.join(cfg.data_path, 'labelsTr')
        # self.test_path = os.path.join(cfg.data_path, 'imagesTs')

        self.train_dict = []
        self.val_dict = []

        self.train_set = None
        self.val_set = None

        self.train_process = None
        self.val_process = None

    def prepare_data(self):
        train_x, train_y = self.get_init()
        for x, y in zip(train_x, train_y):
            info = {'image': x, 'seg': y}
            self.train_dict.append(info)

        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.split_dataset()
        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers)

    def get_preprocess(self):
        cfg = self.cfg
        self.train_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            EnsureChannelFirstd(keys=['image']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"), ),
            Orientationd(keys=['image', 'seg'], axcodes='RAS'),

            RandSpatialCropd(keys=['image', 'seg'], roi_size=cfg.FinalShape, random_size=False),
            SpatialPadd(keys=['image', 'seg'], spatial_size=cfg.FinalShape),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
            ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.l_percent, upper=cfg.u_percent,
                                            b_max=1.0, b_min=0.0, clip=True),

            EnsureTyped(keys=['image', 'seg']),

        ])

        self.val_process = Compose([
            LoadImaged(keys=['image', 'seg']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            EnsureChannelFirstd(keys=['image']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=['image', 'seg'], axcodes='RAS'),

            ScaleIntensityRangePercentilesd(keys=['image'], lower=cfg.l_percent, upper=cfg.u_percent,
                                            b_max=1.0, b_min=0.0, clip=True),
            ToTensord(keys=['image', 'seg']),
        ])

    def get_init(self):
        FP = os.path.join(Config.data_path, 'Brats2021Train')
        train_x, train_y = [], []
        for _, dirs, _ in os.walk(FP):
            for dr in dirs:
                tmp = os.path.join(FP, dr)
                flair_file = glob(os.path.join(tmp, '*flair.nii.gz'), recursive=True)
                t1_file = glob(os.path.join(tmp, '*t1.nii.gz'), recursive=True)
                t1_ce_file = glob(os.path.join(tmp, '*t1ce.nii.gz'), recursive=True)
                t2_file = glob(os.path.join(tmp, '*t2.nii.gz'), recursive=True)
                seg_file = glob(os.path.join(tmp, '*seg.nii.gz'), recursive=True)
                files = [*flair_file, *t1_file, *t1_ce_file, *t2_file]
                train_x.append(files)
                train_y.append(seg_file)

        return train_x, train_y

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_dict)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        if train_num + val_num != num:
            remain = num - train_num - val_num
            val_num += remain

        self.train_dict, self.val_dict = random_split(self.train_dict, [train_num, val_num])


class BatsModel(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(BatsModel, self).__init__()
        self.cfg = cfg
        if cfg.back_bone_name == 'SwinUnet':
            self.net = swinUnet_t_3D(window_size=cfg.window_size, num_classes=cfg.n_classes, in_channel=cfg.in_channels)
        else:
            from monai.networks.nets import UNETR, UNet
            self.net = UNETR(in_channels=cfg.in_channels, out_channels=cfg.n_classes, img_size=cfg.FinalShape)

        self.loss_func = DiceFocalLoss(sigmoid=True, include_background=True)
        # 这个类有bug, 当y_pred和y均为全零矩阵时，此处计算出来的dice系数为nan
        self.metrics = DiceMetric(include_background=True, reduction="mean")

        self.post_pred = Compose([
            transforms.EnsureType(), transforms.Activations(sigmoid=True),
            transforms.AsDiscrete(threshold_values=True, logit_thresh=0.5)])
        self.post_label = Compose([transforms.EnsureType(),
                                   # transforms.AsDiscrete(num_classes=cfg.n_classes)
                                   ])  # label已经是one-hot的形式了

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.parameters(), lr=cfg.lr, eps=1e-7)
        return opt

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['seg'].float()
        y_hat = self.net(x)

        loss, dices = self.shared_step(y_hat, y)
        tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

        self.log('train_tc_dice', tc_dice, prog_bar=True)
        self.log('train_et_dice', et_dice, prog_bar=True)
        self.log('train_wt_dice', wt_dice, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss, 'train_dice': dices}

    def validation_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['seg'].float()
        # 使用滑动窗口进行推理
        y_hat = sliding_window_inference(x, roi_size=cfg.roi_size, overlap=cfg.overlap,
                                         sw_batch_size=1, predictor=self.net)

        loss, dices = self.shared_step(y_hat, y)
        tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

        self.log('valid_tc_dice', tc_dice, prog_bar=True)
        self.log('valid_et_dice', et_dice, prog_bar=True)
        self.log('valid_wt_dice', wt_dice, prog_bar=True)
        self.log('valid_loss', loss, prog_bar=True)

        return {'valid_loss': loss, 'valid_dice': dices}

    def training_epoch_end(self, outputs):
        losses, dices = self.shared_epoch_end(outputs, 'loss', 'train_dice')
        if len(losses) > 0:
            mean_loss = torch.mean(losses, dim=0)
            mean_dice = torch.mean(dices, dim=1)

            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('train_mean_loss', mean_loss, prog_bar=True)
            self.log('tc_train_mean_dice', tc_mean_dice, prog_bar=True)

            self.log('wt_train_mean_dice', wt_mean_dice, prog_bar=True)

            self.log('et_train_mean_dice', et_mean_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, dices = self.shared_epoch_end(outputs, 'valid_loss', 'valid_dice')
        if len(losses) > 0:
            mean_loss = torch.mean(losses)
            mean_dice = torch.mean(dices, dim=1)

            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('valid_mean_loss', mean_loss, prog_bar=True)
            self.log('tc_valid_mean_dice', tc_mean_dice, prog_bar=True)
            self.log('wt_valid_mean_dice', wt_mean_dice, prog_bar=True)
            self.log('et_valid_mean_dice', et_mean_dice, prog_bar=True)

    @staticmethod
    def shared_epoch_end(outputs, loss_key, dice_key):
        losses, dices = [], []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key]
            dice = output[dice_key]

            loss = loss.detach()
            dice = dice.detach()

            losses.append(loss)
            dices.append(dice)

        losses = torch.stack(losses)
        dices = torch.stack(dices)

        dices = torch.permute(dices, (1, 0))

        return losses, dices

    def shared_step(self, y_hat, y):
        # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
        losses = self.loss_func(y_hat, y)
        y_hat = [self.post_pred(i) for i in decollate_batch(y_hat)]
        y = [self.post_label(i) for i in decollate_batch(y)]
        dice = self.metrics(y_hat, y)
        dice = torch.mean(dice, dim=0)
        dices = torch.nan_to_num(dice)

        return losses, dices


data = Brats2021()
model = BatsModel()

early_stop = EarlyStopping(
    monitor='valid_mean_loss',
    patience=5,
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
