# -*- coding: UTF-8 -*-
# import pandas
# RESULT_PATH = r'./simple_fwd_word_lstm_Wed-Dec--8-14:19:47-2021_lr_0.001_nl_1_hs_10_mb_128_bc_20_em_20_rs_5'
import numpy as np
from sklearn.metrics import roc_auc_score
'''
RESULT_PATH = r'./auc_test'
result_list = []

with open(RESULT_PATH, 'r') as f:
    lines = f.readlines() # 把全部数据文件读到一个列表lines中
    for line in lines[1:]:  # 第一行是“batch line second day...” 不要
        list = [float(i) for i in line.strip('\n').split(' ')]
        # print(list)
        result_list.append(list)
        # print(result_list)
        # break

result_array = np.array(result_list)
red = result_array[:, -2]
loss = result_array[:, -1]
print(red)
print(loss)

auc_score = roc_auc_score(red, loss)
print(auc_score) # 0.75
'''

RESULT_PATH = './my_exp/model_result'
import os
for root, dirs, files in os.walk(RESULT_PATH, topdown=True):
    for name in files:
        with open(os.path.join(root, name), 'r') as f:
            result_list = []
            lines = f.readlines()  # 把全部数据文件读到一个列表lines中
            for line in lines[1:]:   # 第一行是“batch line second day...” ,不要
                list = [float(i) for i in line.strip('\n').split(' ')]
                # print(list)
                result_list.append(list)
                # print(result_list)
                # break
            result_array = np.array(result_list)

            red = result_array[:, -2]
            loss = result_array[:, -1]
            auc_score = roc_auc_score(red, loss)
            print("file {} auc_score: {}\n".format(name, auc_score))
