# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
a = tf.test.is_built_with_cuda()
b = tf.test.is_gpu_available()      # 判断GPU是否可以用

print(a)
print(b)