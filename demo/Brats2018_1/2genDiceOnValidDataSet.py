# -*-coding:utf-8-*-
import os

from Brats20183D import *
import pandas as pd


class Config2(Config):
    SavePathDict = {}
    SavePathDict['Unet3D'] = r'./logs\Unet3D\epoch=149-valid_loss=0.25-et_valid_mean_dice=0.74.ckpt'
    SavePathDict['VNet'] = r'./logs\VNet\epoch=69-valid_loss=0.62-et_valid_mean_dice=0.36.ckpt'
    SavePathDict['UNetR'] = r'./logs\UNetR\epoch=42-valid_loss=0.33-et_valid_mean_dice=0.39.ckpt'
    SavePathDict['SwinBTS'] = r'./logs\SwinBTS\epoch=53-valid_loss=0.22-et_valid_mean_dice=0.73.ckpt'
    SavePathDict['TransBTS'] = r'./logs\TransBTS\epoch=51-valid_loss=0.24-et_valid_mean_dice=0.71.ckpt'
    SavePathDict['AttentionUnet'] = r'./logs\AttentionUnet\epoch=49-valid_loss=0.50-et_valid_mean_dice=0.62.ckpt'
    SavePathDict['SwinPureUnet3D'] = r'./logs\SwinPureUnet3D\epoch=38-valid_loss=0.31-et_valid_mean_dice=0.66.ckpt'
    SavePathDict['SwinUnet3D'] = r'./logs\SwinUnet3D\epoch=51-valid_loss=0.22-et_valid_mean_dice=0.73.ckpt'


class Brats2018Model2(Brats2018Model):
    def __init__(self, cfg=Config2()):
        super(Brats2018Model2, self).__init__(cfg)
        self.metrics = [
            DiceMetric(include_background=True,
                       reduction="none"),
            DiceMetric(include_background=True,
                       reduction="none")
        ]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            cfg = self.cfg
            x = batch['image']
            y = batch['label'].float()

            # 使用滑动窗口进行推理
            y_hat = sliding_window_inference(x, roi_size=cfg.roi_size, overlap=cfg.overlap,
                                             sw_batch_size=1, predictor=self.net)
            loss, dices = self.shared_step(y_hat, y, 1)

            return {'valid_loss': loss, 'valid_dice': dices}

    def validation_epoch_end(self, outputs):
        cfg = self.cfg
        losses, mean_dice = self.shared_epoch_end(outputs, 'valid_loss', 1)
        # mean_dice = torch.nan_to_num(mean_dice)

        res_df = pd.DataFrame(columns=['ET', 'TC', 'WT'])
        mean_dice_1 = mean_dice.detach().cpu().numpy()
        res_df['ET'] = mean_dice_1[:, 0]
        res_df['TC'] = mean_dice_1[:, 1]
        res_df['WT'] = mean_dice_1[:, 2]
        if not os.path.exists('./csv_files'):
            os.mkdir('./csv_files')
        res_df.to_csv(os.path.join('./csv_files', cfg.model_name) + '.csv', index=False)


def main(model_name: str = 'Unet3D', seed: int = 3407, cfg=Config2()):
    cfg.model_name = model_name
    cfg.NeedTrain = False
    cfg.seed = seed
    setseed(cfg)

    data = Brats2018DataSet(cfg)
    model = Brats2018Model(cfg)

    early_stop = EarlyStopping(
        monitor='valid_mean_loss',
        patience=5,
    )

    check_point = ModelCheckpoint(dirpath=f'./logs/{cfg.model_name}',
                                  save_last=False,
                                  save_top_k=3, monitor='valid_mean_loss', verbose=True,
                                  filename='{epoch}-{valid_loss:.2f}-{et_valid_mean_dice:.2f}')
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[2],
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

    if cfg.NeedTrain:
        trainer.fit(model, data)
        trainer.save_checkpoint(f'./trained_models/{cfg.model_name}/TrainedModel.ckpt')
    else:
        save_path = cfg.SavePathDict[cfg.model_name]

        argsDict = {'cfg': cfg}
        model = Brats2018Model2.load_from_checkpoint(save_path, **argsDict)  # 这是个类方法，不是对象方法
        model.eval()
        model.freeze()

    # predict_data = Brats2021DataSet()
    # trainer.predict(model, datamodule=predict_data)
    trainer.validate(model, datamodule=data)


if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    g_seeds = [42]

    model_names = ['Unet3D', 'VNet', 'UNetR', 'SwinBTS', 'TransBTS', 'AttentionUnet',
                   'SwinUnet3D', 'SwinPureUnet3D']
    # g_seeds = [2022, 3407]
    # model_names = ['Unet3D', 'SwinUnet3D', 'SwinPureUnet3D', 'UNetR']
    for g_seed in g_seeds:
        for name in model_names:
            main(name, g_seed)
