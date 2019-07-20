#! /usr/bin/python3

'''
【分类】：

导入CIFAR100库学习并分类
'''

import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# 禁用tensorflow通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '2', '3'}

# 宏定义变量
tensorboard_callback = 0
enableLog = False
savePath = 'model_save/keras_rnn_LSTM.h5'
num_words = 1000
max_len = 200

# Tensorboard可视化
def EnableLog():
    logdir="../../Log/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# 加载并处理imdb，通过tf.keras.preprocessing.sequence.pad_sequences函数将每列矩阵都变为长度为max_len
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

## 将每列矩阵都变为长度为max_len
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

## 转换成numpy.array类型矩阵
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Debug
# print(train_len)
# print(x_train[0])

# 选择模式，是否加载已保存的Model
print("Option:")
print("\t1>>New Model")
print("\t2>>Load Model")
print("\t3>>Enbale Log")
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
        # 添加神经网络层，此处使用Dense全连接层，其中参数代表该层神经元数
        # 使用Dropout代表有几率断连，防止神经网络过拟合
        # 使用Enbedding层使单词向量化
        # 使用LSTM层构成RNN
        model.add(tf.keras.layers.Embedding(num_words, 50, input_length=max_len))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(tf.keras.layers.Dense(250, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # 指定模型loss为交叉熵损失函数，优化器使用随机梯度下降优化器，使用监视器监视accuracy情况
        model.compile(loss = tf.losses.binary_crossentropy,
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
    elif(opt == 3):
        EnableLog()
        enableLog = True
        print("Enabled Log")
    else:
        print("No such a option")

while(True):
    try:
        epochs = int(input("Epochs Number: "))
        break
    except:
        print("ERROR INPUT")
        continue
# 训练数据，设置输入数据为x输出数据为y，每次喂入数据大小batch_size，verbose为1表示以进度条
# 方式显示训练进度（0为不显示，2为一行显示一条），总共重复训练epochs次
if(enableLog):
    model.fit(x_train, y_train, 
              batch_size=128, 
              verbose=1, 
              epochs=epochs,
              callbacks=[tensorboard_callback])
else:
    model.fit(x_train, y_train, 
              batch_size=128, 
              verbose=1, 
              epochs=epochs)
# 保存model，若未保存成功则输出错误信息
try:
    model.save(savePath)
    print("Saved Model to:", savePath)
except:
    print("ERROR WHEN SAVE MODEL")

# 上面已经训练好了model，下面通过model.predict()函数，输入x_预测y_，并计算正确答案以检验训练效果
def Predict_X(x_arg_list, list_len = 1):
    i = 0
    for x_arg in x_arg_list:
        i = i + 1
        # 由数组提取一个测试图像并转换成四维矩阵以便Keras神经网络预测
        x_test_ = []
        x_test_.append(x_test[x_arg])
        x_test_ = np.array(x_test_)
        # 得到预测值和真实值的独热编码
        y_predict = model.predict(x_test_)
        y_true = y_test[x_arg]

        # 显示答案
        print('x_test:')
        print(x_test_)
        print('y_predict:')
        print(y_predict)
        print('y_true:')
        print(y_true)
        '''
        plt.subplot(1, list_len, i)
        plt.imshow(x_test_[0, :, :, :])
        predictAns = "Predict: " + str(y_predict) + "\nAnswer: " + str(y_true)
        plt.title(predictAns)
        plt.axis('off')
        '''

while(True):
    print("Option:")
    print("\t1>>Input")
    print("\t2>>Random")
    print("\t3>>Model Summary")
    print("\t0>>Exit")
    m = input('choose: ')
    if(m == '0'):
        exit(1)
    elif(m == '1'):
        # 输入x_arg从test集中选取x_test[x_arg]并调用函数显示预测结果
        print("Input an integer:")
        try:
            x_arg_list = []
            x_arg_list.append(int(input()))
        except:
            print("ERROR INPUT")
            continue
        Predict_X(x_arg_list)
        continue
    elif(m == '2'):
        # 生成list_len个随机数并调用函数显示预测结果
        x_arg_list = []
        test_list_len = len(x_test)
        print("Input random list length:")
        try:
            list_len = int(input())
        except:
            print("ERROR INPUT")
            continue
        for i in range(list_len):
            x_arg_list.append(np.random.randint(test_list_len))
        Predict_X(x_arg_list, list_len=list_len)
        continue
    elif(m == '3'):
        # 打印MODEL的概述
        model.summary()
        continue
