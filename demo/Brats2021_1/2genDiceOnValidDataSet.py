# -*-coding:utf-8-*-
import os

from Brats20213D import *
import pandas as pd


class Config2(Config):
    SavePathDict = {}
    SavePathDict['Unet3D'] = r'./logs/Unet3D/epoch=50-valid_loss=0.15-et_valid_mean_dice=0.83.ckpt'
    SavePathDict['VNet'] = r'./logs/VNet/epoch=105-valid_loss=0.24-et_valid_mean_dice=0.82.ckpt'
    SavePathDict['UNetR'] = r'./logs/UNetR/epoch=50-valid_loss=0.16-et_valid_mean_dice=0.84.ckpt'
    SavePathDict['SwinBTS'] = r'./logs/SwinBTS/epoch=53-valid_loss=0.13-et_valid_mean_dice=0.83.ckpt'
    SavePathDict['TransBTS'] = r'./logs/TransBTS/epoch=58-valid_loss=0.13-et_valid_mean_dice=0.82.ckpt'
    SavePathDict['AttentionUnet'] = r'./logs/AttentionUnet/epoch=50-valid_loss=0.21-et_valid_mean_dice=0.84-v1.ckpt'
    SavePathDict['SwinPureUnet3D'] = r'./logs/SwinPureUnet3D/epoch=52-valid_loss=0.15-et_valid_mean_dice=0.82.ckpt'
    SavePathDict['SwinUnet3D'] = r'./logs/SwinUnet3D/epoch=51-valid_loss=0.12-et_valid_mean_dice=0.83.ckpt'


class Brats2021Model2(Brats2021Model):
    def __init__(self, cfg=Config()):
        super(Brats2021Model2, self).__init__(cfg)
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
            y_hat = sliding_window_inference(x, roi_size=cfg.RoiSize, overlap=cfg.overlap,
                                             sw_batch_size=1, predictor=self.net)
            loss, dices = self.shared_step(y_hat, y, 1)

        # if cfg.SaveTrainPred:
        #     meta_dict = batch['image_meta_dict']  # 将meta_dict中的值转成cpu()向量，原来位于GPU上
        #     for k, v in meta_dict.items():
        #         if isinstance(v, torch.Tensor):
        #             meta_dict[k] = v.detach().cpu()
        #
        #     y_hat = y_hat.detach().cpu()  # 转成cpu向量之后才能存
        #     y_hat = [self.post_trans(i) for i in decollate_batch(y_hat)]
        #     y_hat = [self.label_reverse(i) for i in y_hat]  # 此时y_hat的维度为[H,W,D]
        #     y_hat = torch.stack(y_hat)  # B,H,W,D
        #     y_hat = torch.unsqueeze(y_hat, dim=1)  # 增加通道维度，saver需要的格式为B,C,H,W,D
        #     saver = NiftiSaver(output_dir=self.cfg.ValidSegDir, mode="nearest", print_log=False)
        #     saver.save_batch(y_hat, meta_dict)  # fixme 检查此处用法是否正确

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
    cfg.ValidSegDir = os.path.join(cfg.data_path, 'ValidSeg', model_name)
    cfg.NeedTrain = False
    cfg.seed = seed
    cfg.SaveTrainPred = True
    setseed(cfg)

    data = Brats2021DataSet(cfg)
    model = Brats2021Model(cfg)

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
        model = Brats2021Model2.load_from_checkpoint(save_path, **argsDict)  # 这是个类方法，不是对象方法
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
