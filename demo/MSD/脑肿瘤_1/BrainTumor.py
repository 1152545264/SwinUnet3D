# -*-coding:utf-8-*-
import os
import random
import torch
from torch import nn, functional as F, optim
import monai
import pytorch_lightning as pl
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader, random_split
from glob import glob
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.config import KeysCollection
from torch.utils.data import random_split

from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.data import NiftiSaver, write_nifti
from monai.networks.nets import UNETR, UNet, VNet, DynUNet, SegResNet, SwinUNETR
from monai.networks.nets import TopologySearch

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
    ScaleIntensityRangePercentilesd,
)
from timm.models.layers import trunc_normal_

from SwinUnet_3DV2 import swinUnet_t_3D


def setseed(seed: int = 42):
    pl.seed_everything(seed)
    set_determinism(seed)


def get_nnunet_k_s(final_shape, spacings):  #
    sizes, spacings = final_shape, spacings
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


class Config(object):
    seed = 42  # 设置随机数种子
    spacings = [2.0, 2.0, 1.0]
    # 脑组织窗宽设定为80Hu~100Hu, 窗位为30Hu~40Hu,
    RoiSize = [256 // spacings[0], 256 // spacings[1], 160 // spacings[2]]
    RoiSize = [int(it) for it in RoiSize]
    window_size = [it // 32 for it in RoiSize]  # 针对siwnUnet3D而言的窗口大小,FinalShape[i]能被window_size[i]数整除
    in_channels = 4

    l_percent = 0.5
    u_percent = 99.5

    train_ratio, val_ratio, test_ratio = [0.8, 0.2, 0.0]
    BatchSize = 1
    NumWorkers = 4  # 如果此处NumWorkers > 0, 则需要加大操作系统中swap分区(Linux)的数值或者虚拟内存的数值(windows)

    max_epoch = 60
    min_epoch = 50

    LRCycle = 10
    '''
    标签中绿色为浮肿区域(ED,peritumoral edema) (标签2)、黄色为增强肿瘤区域(ET,enhancing tumor)(标签4)、
    红色为坏疽(NET,non-enhancing tumor)(标签1)、背景(标签0)
    WT = ED + ET + NET
    TC = ET+NET
    ET
    '''
    n_classes = 3  # 分割总的类别数，在三个通道上分别进行前景和背景的分割，三个通道为：TC（肿瘤核心）、ET（肿瘤增强)和WT（整个肿瘤）。

    lr = 3e-4  # 学习率

    # 滑动窗口推理时使用
    roi_size = RoiSize
    overlap = 0.0

    # model_name = 'SwinUnet3D'
    # model_name = 'Unet3D'
    # model_name = 'VNet'
    # model_name = 'UNetR'
    model_name = 'SwinUNETR'

    ModelDict = {}
    ArgsDict = {}
    ModelDict['Unet3D'] = UNet
    ArgsDict['Unet3D'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                          'channels': (32, 64, 128, 256, 512), 'strides': (2, 2, 2, 2)}

    ModelDict['VNet'] = VNet
    ArgsDict['VNet'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes, 'dropout_prob': 0.0, }

    ModelDict['UNetR'] = UNETR
    ArgsDict['UNetR'] = {'in_channels': in_channels, 'out_channels': n_classes, 'img_size': RoiSize}

    ModelDict['SwinUNETR'] = SwinUNETR
    ArgsDict['SwinUNETR'] = {'in_channels': in_channels, 'out_channels': n_classes, 'img_size': RoiSize}

    kernels, strides = get_nnunet_k_s(RoiSize, spacings)
    ModelDict['nn-unet'] = DynUNet
    ArgsDict['nn-unet'] = {'spatial_dims': 3, 'in_channels': in_channels, 'out_channels': n_classes,
                           'kernel_size': kernels, 'strides': strides, 'upsample_kernel_size': strides[1:],
                           'norm_name': 'instance', 'deep_supervision': True, 'deep_supr_num': 3}

    ModelDict['SwinUnet3D'] = swinUnet_t_3D
    ArgsDict['SwinUnet3D'] = {'in_channel': in_channels, 'num_classes': n_classes, 'window_size': window_size}

    # NeedTrain = False
    NeedTrain = True
    SaveTrainPred = True
    data_path = r'D:\Caiyimin\Dataset\MSD\BrainTumor'
    ValidSegDir = os.path.join(data_path, 'ValidSeg', model_name)
    PredDataDir = os.path.join(data_path, 'Brats2021Pred')
    PredSegDir = os.path.join(data_path, 'PredSeg', model_name)


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


class ReverseBratsLabelNumpy(transforms.Transform):
    def __call__(self, pred):
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        pred = pred.astype(np.bool)
        tc = pred[0, ::]
        wt = pred[1, ::]
        et = pred[2, ::]
        res = np.zeros(pred.shape[1:])

        # label_4 = et
        # label_2 = wt - tc
        # label_1 = tc - et

        label_4 = et
        label_2 = np.logical_and(wt, np.logical_not(tc))
        label_1 = np.logical_and(tc, np.logical_not(et))

        res[label_1] = 1
        res[label_2] = 2
        res[label_4] = 4
        return res


class ReverseBratsLabel(transforms.Transform):
    def __call__(self, pred):
        pred = pred.bool()
        tc = pred[0, ::]
        wt = pred[1, ::]
        et = pred[2, ::]
        res = torch.zeros(pred.shape[1:])

        # label_4 = et
        # label_2 = wt - tc
        # label_1 = tc - et

        label_4 = et
        label_2 = torch.logical_and(wt, torch.logical_not(tc))
        label_1 = torch.logical_and(tc, torch.logical_not(et))

        res[label_1] = 1
        res[label_2] = 2
        res[label_4] = 4
        return res


def save_pred(pred, predSavePath):
    pred = pred.detach().cpu().numpy()

    pred2 = ReverseBratsLabel()(pred)
    write_nifti(pred2, file_name=predSavePath)


class Brats2021DataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(Brats2021DataSet, self).__init__()
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.train_path = os.path.join(cfg.data_path, 'imagesTr')
        self.label_tr_path = os.path.join(cfg.data_path, 'labelsTr')
        self.test_path = os.path.join(cfg.data_path, 'imagesTs')

        self.train_dict = []
        self.val_dict = []

        self.train_set = None
        self.val_set = None

        self.train_process = None
        self.val_process = None

        self.pred_dir = cfg.PredDataDir
        self.predSaveDir = cfg.PredSegDir
        self.pred_files = []
        self.pred_set = None
        self.test_transforms = None

    def prepare_data(self):
        train_x, train_y = self.initTrainVal()
        for x, y in zip(train_x, train_y):
            info = {'image': x, 'label': y}
            self.train_dict.append(info)

        self.init_predFiles()

        self.get_preprocess()

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.split_dataset()

        self.train_set = Dataset(self.train_dict, transform=self.train_process)
        self.val_set = Dataset(self.val_dict, transform=self.val_process)
        self.pred_set = Dataset(self.pred_files, self.test_transforms)

    def train_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.train_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers, shuffle=True)

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.val_set, batch_size=cfg.BatchSize, num_workers=cfg.NumWorkers, shuffle=False)

    def predict_dataloader(self):
        cfg = self.cfg
        return DataLoader(self.pred_set, batch_size=cfg.BatchSize,
                          num_workers=cfg.NumWorkers, shuffle=False)

    def get_preprocess(self):
        cfg = self.cfg
        self.train_process = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=cfg.spacings,
                         mode=("bilinear", "nearest"), ),

                # RandSpatialCropd(keys=["image", "label"], roi_size=cfg.FinalShape, random_size=False),
                # SpatialPadd(keys=['image', 'label'], spatial_size=cfg.FinalShape),

                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

                NormalizeIntensityd(keys="image", nonzero=True,
                                    channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.val_process = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=cfg.spacings,
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True,
                                    channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.test_transforms = Compose([
            LoadImaged(keys='image'),
            EnsureChannelFirstd(keys='image'),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys='image', axcodes="RAS"),
            NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
            EnsureTyped(keys='image'),
        ])

    def initTrainVal(self):
        train_x = glob(os.path.join(self.train_path, '*.nii.gz'))
        train_y = glob(os.path.join(self.label_tr_path, '*.nii.gz'))

        return train_x, train_y

    def init_predFiles(self):
        test_x = glob(os.path.join(self.test_path, '*.nii.gz'))
        for file in test_x:
            pred_path = self.pred_dir
            info_key = {'image': file, 'pred_path': pred_path}
            self.pred_files.append(info_key)

    def split_dataset(self):
        cfg = self.cfg
        num = len(self.train_dict)
        train_num = int(num * cfg.train_ratio)
        val_num = int(num * cfg.val_ratio)
        if train_num + val_num != num:
            remain = num - train_num - val_num
            val_num += remain

        self.train_dict, self.val_dict = random_split(self.train_dict, [train_num, val_num],
                                                      generator=torch.Generator().manual_seed(cfg.seed))
        # 设置generator保证不同模型划分出来的训练集和验证集相同


class Brats2021Model(pl.LightningModule):
    def __init__(self, cfg=Config()):
        super(Brats2021Model, self).__init__()
        self.cfg = cfg
        model = cfg.ModelDict[cfg.model_name]
        kwargs = cfg.ArgsDict[cfg.model_name]
        self.net = model(**kwargs)
        if cfg.model_name not in ('SwinUnet3D', 'LKA_Unet3D'):  # Monai中的模型缺乏参数初始化，不加参数初始化容易导致某些通道的dice系数爆零
            ModelParamInit(self.net)

        self.loss_func = DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                                  squared_pred=True, to_onehot_y=False,
                                  sigmoid=True, )
        self.metrics = [
            DiceMetric(include_background=True,
                       reduction="mean_batch"),
            DiceMetric(include_background=True,
                       reduction="mean_batch")
        ]
        # self.HD_metric = HausdorffDistanceMetric(include_background=True, reduction='mean_batch')
        self.post_trans = Compose([EnsureType(),
                                   Activations(sigmoid=True),
                                   AsDiscrete(threshold_values=True)])
        self.label_reverse = ReverseBratsLabel()

    def configure_optimizers(self):
        cfg = self.cfg
        opt = optim.AdamW(params=self.net.parameters(),
                          lr=cfg.lr, eps=1e-7,
                          weight_decay=1e-5)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.LRCycle)
        return {'optimizer': opt, 'lr_scheduler': lr_scheduler, 'monitor': 'valid_mean_loss'}

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        x = batch['image']
        y = batch['label'].float()
        y_hat = sliding_window_inference(x, roi_size=cfg.roi_size,
                                         overlap=cfg.overlap,
                                         sw_batch_size=1,
                                         predictor=self.net)

        loss, dices = self.shared_step(y_hat, y, 0)
        tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

        self.log('train_tc_dice', tc_dice, prog_bar=True)
        self.log('train_et_dice', et_dice, prog_bar=True)
        self.log('train_wt_dice', wt_dice, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss, 'train_dice': dices}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            cfg = self.cfg
            x = batch['image']
            y = batch['label'].float()

            # 使用滑动窗口进行推理
            y_hat = sliding_window_inference(x, roi_size=cfg.roi_size, overlap=cfg.overlap,
                                             sw_batch_size=1, predictor=self.net)
            loss, dices = self.shared_step(y_hat, y, 1)
            tc_dice, wt_dice, et_dice = dices[0], dices[1], dices[2]

            self.log('valid_tc_dice', tc_dice, prog_bar=True)
            self.log('valid_et_dice', et_dice, prog_bar=True)
            self.log('valid_wt_dice', wt_dice, prog_bar=True)

            self.log('valid_loss', loss, prog_bar=True)
            if wt_dice > 0.85 and self.cfg.SaveTrainPred:  # 保存验证过程中的预测标签
                meta_dict = batch['image_meta_dict']  # 将meta_dict中的值转成cpu()向量，原来位于GPU上
                for k, v in meta_dict.items():
                    if isinstance(v, torch.Tensor):
                        meta_dict[k] = v.detach().cpu()

                y_hat = y_hat.detach().cpu()  # 转成cpu向量之后才能存
                y_hat = [self.post_trans(i) for i in decollate_batch(y_hat)]
                y_hat = [self.label_reverse(i) for i in y_hat]  # 此时y_hat的维度为[H,W,D]
                y_hat = torch.stack(y_hat)  # B,H,W,D
                y_hat = torch.unsqueeze(y_hat, dim=1)  # 增加通道维度，saver需要的格式为B,C,H,W,D
                saver = NiftiSaver(output_dir=self.cfg.ValidSegDir, mode="nearest", print_log=False)
                saver.save_batch(y_hat, meta_dict)  # fixme 检查此处用法是否正确

                # names = batch['image_meta_dict']['filename_or_obj']
                # self.save_pred_label(y_hat, names)

            return {'valid_loss': loss, 'valid_dice': dices}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            cfg = self.cfg
            img = batch['image']
            preds = sliding_window_inference(img, roi_size=cfg.RoiSize, overlap=cfg.overlap,
                                             sw_batch_size=1, predictor=self.net)
            meta_dict = batch['image_meta_dict']
            for k, v in meta_dict.items():
                if isinstance(v, torch.Tensor):
                    meta_dict[k] = v.detach().cpu()

            preds = preds.detach().cpu()
            preds = [self.post_trans(i) for i in decollate_batch(preds)]
            preds = [self.label_reverse(i) for i in preds]
            preds = torch.stack(preds)  # B,H,W,D
            preds = torch.unsqueeze(preds, dim=1)  # 增加通道维度，saver需要的格式为B,C,H,W,D

            saver = NiftiSaver(output_dir=self.cfg.PredSegDir, mode="nearest")
            saver.save_batch(preds, meta_dict)

    def training_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'loss')
        if len(losses) > 0:
            mean_loss = torch.mean(losses)
            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('train_mean_loss', mean_loss, prog_bar=True)
            self.log('tc_train_mean_dice', tc_mean_dice, prog_bar=True)
            self.log('wt_train_mean_dice', wt_mean_dice, prog_bar=True)
            self.log('et_train_mean_dice', et_mean_dice, prog_bar=True)

    def validation_epoch_end(self, outputs):
        losses, mean_dice = self.shared_epoch_end(outputs, 'valid_loss', 1)
        if len(losses) > 0:
            mean_loss = torch.mean(losses)

            # 三个通道为：TC（肿瘤核心）、WT（整个肿瘤）和ET（肿瘤增强)。
            tc_mean_dice, wt_mean_dice, et_mean_dice = mean_dice[0], mean_dice[1], mean_dice[2]

            self.log('valid_mean_loss', mean_loss, prog_bar=True)
            self.log('tc_valid_mean_dice', tc_mean_dice, prog_bar=True)
            self.log('wt_valid_mean_dice', wt_mean_dice, prog_bar=True)
            self.log('et_valid_mean_dice', et_mean_dice, prog_bar=True)

    def shared_epoch_end(self, outputs, loss_key, stage: int = 0):  # 0为训练阶段，1为验证阶段
        losses = []
        for output in outputs:
            # loss = output['loss'].detach().cpu().numpy()
            loss = output[loss_key]
            loss = loss.detach()
            losses.append(loss)

        losses = torch.stack(losses)

        # if stage == 0:
        #     mean_dice = self.train_metric.aggregate()
        #       self.train_metric.reset()
        # else:
        #     mean_dice = self.val_metric.aggregate()
        #     self.val_metric.reset()
        mean_dice = self.metrics[stage].aggregate()

        self.metrics[stage].reset()

        return losses, mean_dice

    def shared_step(self, y_hat, y, stage: int = 0):  # 0为训练阶段，1为验证阶段
        loss = self.loss_func(y_hat, y)

        y_hat = [self.post_trans(i) for i in decollate_batch(y_hat)]

        # if stage == 0:
        #     dice = self.train_metric(y_pred=y_hat, y=y)
        # else:
        #     dice = self.val_metric(y_pred=y_hat, y=y)
        dice = self.metrics[stage](y_pred=y_hat, y=y)

        loss = torch.nan_to_num(loss)
        dice = torch.nan_to_num(dice)
        dice = torch.mean(dice, dim=0)

        return loss, dice


def ModelParamInit(model):
    assert isinstance(model, nn.Module)
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def main():
    data = Brats2021DataSet()
    model = Brats2021Model()

    early_stop = EarlyStopping(
        monitor='valid_mean_loss',
        patience=5,
    )

    cfg = Config()
    check_point = ModelCheckpoint(dirpath=f'./logs/{cfg.model_name}',
                                  save_last=False,
                                  save_top_k=3, monitor='valid_mean_loss', verbose=True,
                                  filename='{epoch}-{valid_loss:.2f}-{et_valid_mean_dice:.2f}')
    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        reload_dataloaders_every_n_epochs=1000,
        max_epochs=cfg.max_epoch,
        min_epochs=cfg.min_epoch,
        gpus=1,
        # auto_select_gpus=True, # 这个参数针对混合精度训练时，不能使用
        # auto_lr_find=True,
        auto_scale_batch_size=True,
        logger=TensorBoardLogger(save_dir=f'./logs', name=f'{cfg.model_name}'),
        callbacks=[early_stop, check_point],
        precision=16,
        accumulate_grad_batches=16,
        num_sanity_val_steps=0,
        log_every_n_steps=400,
        gradient_clip_val=1e3,
        gradient_clip_algorithm='norm',
    )

    if Config.NeedTrain:
        trainer.fit(model, data)
        trainer.save_checkpoint(f'./trained_models/{cfg.model_name}/TrainedModel.ckpt')
    else:
        save_path = f'./trained_models/{cfg.model_name}/epoch=95-valid_loss=0.15-et_valid_mean_dice=0.85.ckpt'

        model = Brats2021Model.load_from_checkpoint(save_path)  # 这是个类方法，不是对象方法
        model.eval()
        model.freeze()

    predict_data = Brats2021DataSet()
    trainer.predict(model, datamodule=data)


if __name__ == '__main__':
    setseed()
    main()
