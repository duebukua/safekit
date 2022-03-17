# -*- coding: UTF-8 -*-
import tensorflow as tf
# print(tf.__version__)
import numpy as np
import json
import sys
import os

from safekit.batch import OnlineBatcher
from safekit.graph_training_utils import ModelRunner, EarlyStop
from safekit.tf_ops import lm_rnn
from safekit.util import get_mask, Parser

tf.set_random_seed(408)
np.random.seed(408)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

layer_list = [10]
lr = 1e-3
embed_size = 20
mb_size = 128

maxbadcount = 20    # Threshold for early stopping.'

dataspecs = json.load(open('../features/specs/lm/lanl_word_config.json', 'r'))
sentence_length = dataspecs['sentence_length'] - 1
token_set_size = dataspecs['token_set_size']
# -------------------------------------------------------------------------------------------------------------------
# 输入
x = tf.placeholder(tf.int32, [None, sentence_length])
t = tf.placeholder(tf.int32, [None, sentence_length])
ph_dict = {'x': x, 't': t}

# -------------------------------------------------------------------------------------------------------------------
token_embed = tf.Variable(tf.truncated_normal([token_set_size, embed_size]))  # 词向量矩阵

# 网络中的数据流向，计算一个小批量的loss，批量大小取决于x的维度
token_losses, h_states, final_h = lm_rnn(x, t, token_embed, layer_list)
line_losses = tf.reduce_mean(token_losses, axis=1)
avg_loss = tf.reduce_mean(line_losses)

# -------------------------------------------------------------------------------------------------------------------
# 定义输出
outfile = open('results', 'w')  # 测试阶段的loss输出到results中
outfile.write("batch line second day user red loss\n")

def write_results(data_dict, loss, outfile, batch):
    for n, s, d, u, r, l in zip(data_dict['line'].flatten().tolist(),
                                data_dict['second'].flatten().tolist(),
                                data_dict['day'].flatten().tolist(),
                                data_dict['user'].flatten().tolist(),
                                data_dict['red'].flatten().tolist(),
                                loss.flatten().tolist()):
        outfile.write('%s %s %s %s %s %s %r\n' % (batch, int(n), int(s), int(d), int(u), int(r), l))
# -------------------------------------------------------------------------------------------------------------------
model = ModelRunner(avg_loss, ph_dict, learnrate=lr)

eval_tensors = [avg_loss, line_losses]

# -------------------------------------------------------------------------------------------------------------------
def trainday(is_training, f):  # 一天的数据来训练
    batch_num = 0
    # data = OnlineBatcher('/home/hutch_research/data/lanl/char_feats/word_day_split/' + f, mb_size, delimiter=' ')
    data = OnlineBatcher('/home/chengy/MyCode/safekit/data_examples/lanl/lm_feats/word_day_split/' + f, mb_size, delimiter=' ')
    raw_batch = data.next_batch()
    cur_loss = sys.float_info.max
    check_error = EarlyStop(maxbadcount)
    endx = raw_batch.shape[1] - 1
    endt = raw_batch.shape[1]
    training = check_error(raw_batch, cur_loss)  # 没有太多的badcase代表模型并没有急剧恶化
    while training:
        data_dict = {'line': raw_batch[:, 0], 'second': raw_batch[:, 1],
                     'day': raw_batch[:, 2], 'user': raw_batch[:, 3],
                     'red': raw_batch[:, 4], 'x': raw_batch[:, 5:endx],
                     't': raw_batch[:, 6:endt]}

        _, cur_loss, pointloss = model.train_step(data_dict, eval_tensors, update=is_training)
        if not is_training:
            write_results(data_dict, pointloss, outfile, batch_num)
        batch_num += 1

        print('%s %s %s %s %s %s %r' % (raw_batch.shape[0], data_dict['line'][0],
                                        data_dict['second'][0], ('fixed', 'update')[is_training],
                                        f, data.index, cur_loss))

        raw_batch = data.next_batch()
        training = check_error(raw_batch, cur_loss)
        if training < 0:
            exit(0)

# -------------------------------------------------------------------------------------------------------------------

files = dataspecs['test_files']

for idx, f in enumerate(files[:-1]):
    trainday(True, f)
    trainday(False, files[idx + 1])
outfile.close()