#! /usr/bin/python3

'''
【识别手写数字】：

导入MNIST库学习并识别手写数字
'''

import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 禁用tensorflow通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '2', '3'}

# 定义保存路径
savePath = 'model_save/keras_image_handwrite_number.h5'

# 加载并处理MINST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 转换成单通道图像
img_rows, ima_cols = x_train[0].shape[0], x_train[0].shape[1]
x_train = x_train.reshape(x_train.shape[0], img_rows, ima_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, ima_cols, 1)
# 转换成numpy.array类型矩阵
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# 输入数据标准化
x_train = x_train.astype(np.float)/255
x_test = x_test.astype(np.float)/255
# 输出数据进行独热编码处理
train_len = len(set(y_train))
y_train = tf.keras.utils.to_categorical(y_train, train_len)
y_test = tf.keras.utils.to_categorical(y_test, train_len)

# Debug
# print(train_len)
# print(x_train[0])

# 选择模式，是否加载已保存的Model
print("Option:")
print("\t1>>New Model")
print("\t2>>Load Model")
print("\t0>>Exit")

while(True):
    try:
        opt = int(input("Choose:"))
    except:
        print("ERROR INPUT")
        continue

    if(opt == 0):
        exit(1)
    elif(opt == 1):
        # 如果没有已保存的则自己创建
        # 使用Sequential模型
        model = tf.keras.Sequential()
        # 添加神经网络层，此处使用Dense全连接层，其中参数代表输出数
        # 使用Dropout代表有几率断连，防止神经网络过拟合
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(train_len, activation='softmax'))


        # 指定模型loss，优化器使用随机梯度下降优化器，使用监视器监视accuracy情况
        model.compile(loss = tf.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])
        print("Created")
        break
    elif(opt == 2):
        try:
            # 加载已保存的model
            model = tf.keras.models.load_model(savePath)
            print("Loaded Model")
            break
        except:
            print("No Model Found")
            continue
    else:
        print("No such a option")

while(True):
    try:
        loop_n = int(input("Fit Loop Number: "))
        break
    except:
        print("ERROR INPUT")
        continue
for loop in range(loop_n):
    print("Loop %d/%d:" %(loop, loop_n))
    # 训练数据，设置输入数据为x输出数据为y，每次喂入数据大小batch_size，verbose为1表示以进度条
    # 方式显示训练进度（0为不显示，2为一行显示一条），总共重复训练epochs次
    model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=10)
    # 保存model，若未保存成功则输出错误信息
    try:
        model.save(savePath)
        print("Saved Model to:", savePath)
    except:
        print("ERROR WHEN SAVE MODEL")

# 上面已经训练好了model，下面通过model.predict()函数，输入x_预测y_，并计算正确答案以检验训练效果
def Predict_X(x_num):
    y_predict = model.predict(x_test[x_num])
    y_true = y_test[x_num]
    print("Predict:")
    print("y_predict = ")
    print(y_predict)
    print("y_true = ")
    print(y_true)

while(True):
    print("Option:")
    print("\t1>>Input")
    print("\t2>>Model Summary")
    print("\t0>>Exit")
    m = input('choose: ')
    if(m == '0'):
        exit(1)
    elif(m == '1'):
        print("Input an integer:")
        try:
            x_num = int(input())
        except:
            print("ERROR INPUT")
            continue
        Predict_X(x_num)
        continue
    elif(m == '2'):
        # 打印MODEL的概述
        model.summary()
        continue
