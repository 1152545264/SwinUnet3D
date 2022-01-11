# -*-coding:utf-8-*-
import pytorch_lightning as pl
from monai import transforms
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from monai.config import KeysCollection
from monai.utils import set_determinism

pl.seed_everything(42)
set_determinism(42)


class Config(object):
    pass


class ObserveShape(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ObserveShape, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # 输入是(X,Y,Z)
        return d


# 适用于分割有重叠的部分
class ConvertLabeld(transforms.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super(ConvertLabeld, self).__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            res = []
            # 将 tumor和pancreas合并成pancreas
            res.append(np.logical_or(img == 1, img == 2))
            res.append(img == 2)  # tumor通道

            res = np.stack(res, axis=0)
            # res = np.concatenate(res, axis=0)
            res = res.astype(np.float)
            d[key] = res
        return d


class LitsDataSet(pl.LightningDataModule):
    def __init__(self, cfg=Config()):
        super(LitsDataSet, self).__init__()
        pass

    def prepare_data(self):
        self.get_init()
        pass

    # 划分训练集，验证集，测试集以及定义数据预处理和增强，
    def setup(self, stage=None) -> None:
        self.split_dataset()
        self.get_preprocess()
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    # 定义训练集和测试集的transformer，包括读取数据，数据增强，像素体素归一化等等
    def get_preprocess(self):
        pass

    def get_init(self):
        pass

    def split_dataset(self):
        pass


class Lung(pl.LightningModule):
    # 定义网络模型，损失函数类，metrics类以及后处理标签函数等
    def __init__(self, cfg=Config()):
        super(Lung, self).__init__()
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    # training_epoch_end,valid_epoch_end,test_epoch_end共同步骤可写在此函数中
    def shared_epoch_end(self, outputs, loss_key):
        pass

    # training_step,valid_step,test_step共同步骤可写在此函数中
    def shared_step(self, y_hat, y):
        pass


data = LitsDataSet()
model = Lung()

early_stop = EarlyStopping()

cfg = Config()
check_point = ModelCheckpoint()
trainer = pl.Trainer(
    progress_bar_refresh_rate=10,
    gpus=1,
    # auto_select_gpus=True, # 这个参数针对混合精度训练时，不能使用

    # auto_lr_find=True,
    auto_scale_batch_size=True,
    callbacks=[early_stop, check_point],
    precision=16,  # 16为指定半精度训练，
    accumulate_grad_batches=4,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    auto_lr_find=True
)
trainer.fit(model, data)
# -*-coding:utf-8-*-
