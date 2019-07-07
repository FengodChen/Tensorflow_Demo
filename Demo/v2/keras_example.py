#! /usr/bin/python3

import tensorflow as tf
import numpy as np

# 开始搭建神经网络

# 使用Sequential模型
model = tf.keras.Sequential()
# 添加神经网络层，此处使用Dense全连接层，输入节点数为1，输出节点数为1
model.add(tf.keras.layers.Dense(units = 1, input_dim = 1))
# 指定模型loss为最小二乘误差mse，优化器使用随机梯度下降优化器sgd
model.compile(loss='mse', optimizer='sgd')