#! /usr/bin/python3

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# 使用Sequential模型
model = tf.keras.Sequential()
# 添加神经网络层，此处使用Dense全连接层，输入节点数为1，输出节点数为1
model.add(layers.Dense(units = 5, input_dim = 5))
# 指定模型loss为最小二乘误差mse，优化器使用随机梯度下降优化器sgd，使用监视器监视accuracy情况
model.compile(loss = tf.losses.mse,
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

# 随机生成输入x和输出y，这里令x为1*5矩阵，总共100000个数据，输出y=x*0.6
x = np.random.random((100000,5))*100
y = x*0.6

# 训练数据，设置输入数据为x输出数据为y，每次喂入数据大小为32，verbose为1表示以进度条
# 方式显示训练进度（0为不显示，2为一行显示一条），总共重复训练10次
model.fit(x, y, batch_size=32, verbose=1, epochs=10)

# 上面已经训练好了model，下面通过model.predict()函数，输入x_预测y_，并计算正确答案以检验训练效果
while(True):
    x_ = np.random.random((1, 5))*100
    y_ = model.predict(x_)
    y_t = x_*0.6
    print("Predict:")
    print("x = ")
    print(x_)
    print("y_predict = ")
    print(y_)
    print("y_true = ")
    print(y_t)
    input()
