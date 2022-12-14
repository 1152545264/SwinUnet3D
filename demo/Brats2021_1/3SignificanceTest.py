# -*-coding:utf-8-*-
import numpy as np
import pandas
import numpy
import os, glob

import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
from scipy import stats

columns = ['ET', 'TC', 'WT']
compare_models = ['Unet3D', 'VNet', 'UNetR', 'TransBTS', 'SwinBTS', 'AttentionUnet',
                  'SwinPureUnet3D']
target_model = 'SwinUnet3D'

target_content = pd.read_csv(os.path.join(r'./csv_files', target_model + '.csv'))

for c in columns:
    for source_model in compare_models:
        source_content = pd.read_csv(os.path.join(r'./csv_files', source_model + '.csv'))
        arr1 = source_content[c]
        arr2 = target_content[c]

        arr1 = arr1.dropna()
        arr2 = arr2.dropna()
        assert arr1.__len__() == arr2.__len__()

        # res = ttest_ind(arr1, arr2, ).pvalue # 独立样本t检验
        res = ttest_rel(arr1, arr2, ).pvalue  # 配对样本t检验
        res = round(res, 4)

        res2 = stats.wilcoxon(arr1, arr2).pvalue
        res2 = round(res2, 4)

        # res3 = stats.ks_2samp(arr1, arr2).pvalue
        # res3 = round(res3, 4)

        # print(c, '*****', '{0:15} **** {1:15}'.format(source_model, target_model),
        #       "***", '{0:6} **** {1:6} **** {2:6}'.format(res, res2, res3))

        print(c, '*****', '{0:15} **** {1:15}'.format(source_model, target_model),
              "***", '{0:6} **** {1:6}'.format(res, res2))
    print()
