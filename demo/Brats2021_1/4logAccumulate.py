# -*-coding:utf-8-*-
import os

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm import tqdm
from glob import glob
import os


def getEventFiles(log_path: str = r'./logs'):
    event_files = glob(os.path.join(log_path, "*/*/events.out.tfevents*"))
    # print(event_files)
    return event_files


def getAllModelNames():
    model_names = []
    eventFiles = getEventFiles()
    for e_f in eventFiles:
        # 解析模型名字
        model_name = e_f.split('\\')[1]
        model_names.append(model_name)
    return model_names


def getKeyDF(TargetKeys):
    keyDicts = {}

    model_names = getAllModelNames()
    for t_k in TargetKeys:
        columns = ['steps'] + model_names
        df = pd.DataFrame(columns=columns, copy=True)
        keyDicts[t_k] = df
    return keyDicts


def Merge(df: pd.DataFrame, right: np.array, key):
    right = pd.Series(right)
    x, y = df[key].size, right.size
    if x < y:  # 先插入一堆空行或者重新设置df的index
        n = df.columns
        n = len(n)
        for i in range(y - x):
            df = pd.concat([df, pd.DataFrame([[np.NAN] * n],
                                             columns=df.columns)], ignore_index=True)
        # df = df.reindex(labels=range(y))
    df[key] = right
    return df


if __name__ == '__main__':
    TargetKeys = ['valid_mean_loss', 'tc_valid_mean_dice',
                  'wt_valid_mean_dice', 'et_valid_mean_dice']

    save_path = './graphTable'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    key2Dicts = getKeyDF(TargetKeys)

    eventFiles = getEventFiles()

    for t_k in TargetKeys:
        df = key2Dicts[t_k]
        for e_f in eventFiles:
            event_data = event_accumulator.EventAccumulator(e_f)
            event_data.Reload()
            tmp = event_data.Scalars(t_k)
            tmp = pd.DataFrame(tmp).values

            steps = df['steps']
            x, y = steps.size, tmp.shape[0]
            if x == 0 or x < y:
                df = Merge(df, tmp[:, 1], 'steps')

            # 解析模型名字
            model_name = e_f.split('\\')[1]
            df[model_name] = pd.Series(tmp[:, 2])
            # df = Merge(df, tmp[:, 2], model_name)

        key2Dicts[t_k] = df
        df.to_csv(os.path.join(save_path, t_k + '.csv'), index=False)
