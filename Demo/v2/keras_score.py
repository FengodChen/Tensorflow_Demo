#! /usr/bin/python3
'''
【是否通过问题】：

假设某学校需要考5门科目，总分是按照各科一定权重得到
如果总分大于某成绩，则通过，反正则没通过。

现在你只知道五科的成绩和是否通过，通过深
度学习，根据五科成绩预测是否通过。
'''
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

# 禁用tensorflow通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '2', '3'}

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

        # 此层用来学习权重
        model.add(layers.Dense(5))
        model.add(layers.Dropout(0.2))
        # 此层用来适应激活函数
        model.add(layers.Dense(5))
        model.add(layers.Dropout(0.2))
        # 此层调用激活函数逼近得到二值化结果
        model.add(layers.Dense(1, activation='sigmoid'))
        #model.add(layers.Dense(1, activation='tanh'))
        #model.add(layers.Dense(1, activation='relu'))

        # 指定模型loss为最小二乘误差mse，优化器使用随机梯度下降优化器sgd，使用监视器监视accuracy情况
        model.compile(loss = 'mse', # list: ['mae', 'mse']
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])
        print("Created")
        break
    elif(opt == 2):
        try:
            # 加载已保存的model
            model = tf.keras.models.load_model('model_save/keras_score.h5')
            print("Loaded Model")
            break
        except:
            print("No Model Found")
            continue
    else:
        print("No such a option")

# 随机生成输入x和输出y
def f(x):
    w_ = np.array([0.5, 0.15, 0.35, 0.05, 0.05])
    # 矩阵乘积
    y = np.matmul(x, w_)
    # 矩阵转置
    y = y.reshape(y.size, 1)
    # 对y中每一个元素，若>60则为1，反之则为0
    return np.where(y > 60, 1, 0)

while(True):
    try:
        loop_n = int(input("Fit Loop Number: "))
        break
    except:
        print("ERROR INPUT")
        continue
for loop in range(loop_n):
    print("Loop %d:" %loop)
    # 正态分布矩阵，mu=0.6，sigma=0.3，形状为(100000*5)
    x = np.random.normal(0.6, 0.3, (100000,5))*100
    # 因为成绩范围为0到100，所以将其控制在0到100范围内
    x = np.where(x > 100, 100, x)
    x = np.where(x < 0, 0, x)
    # 因为成绩为整数，故转成整数，因为feed_dict只能是float型，故再转成float
    x = x.astype(np.int8).astype(np.float)
    # 训练数据，设置输入数据为x输出数据为y，每次喂入数据大小batch_size，verbose为1表示以进度条
    # 方式显示训练进度（0为不显示，2为一行显示一条），总共重复训练epochs次
    model.fit(x, f(x), batch_size=32, verbose=1, epochs=10)
    # 保存model至./model_save/keras_pro.h5
    # 若未保存成功则输出错误信息
    try:
        model.save('model_save/keras_score.h5')
        print("Saved Model to './model_save/keras_score.h5'")
    except:
        print("ERROR WHEN SAVE MODEL")

# 打印MODEL的概述
model.summary()

# 上面已经训练好了model，下面通过model.predict()函数，输入x_预测y_，并计算正确答案以检验训练效果
while(True):
    m = input()
    if(m == '0'):
        break
    x_ = np.random.normal(0.6, 0.3, (1,5))*100
    x_ = np.where(x_ > 100, 100, x_)
    x_ = np.where(x_ < 0, 0, x_)
    x_ = x_.astype(np.int8).astype(np.float)
    y_ = (model.predict(x_))[0, 0]
    y_t = f(x_)[0, 0]
    print("Predict:")
    print("x = ")
    print(x_)
    print("y_predict = ")
    print(y_)
    print("y_true = ")
    print(y_t)
