#! /usr/bin/python3
import sys
sys.path.append("./function")

import tensorflow as tf
import TrainData

# disable eager in tensorflow v2
tf.compat.v1.disable_eager_execution()

# 定义数据
myLen = 50
w_true = TrainData.getRandomList(myLen, 1)

# 已知的输入输出
x_train = tf.compat.v1.placeholder(shape=[myLen], dtype=tf.float32)
y_train = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)

# 定义权重，预测输出函数，误差，训练方向
w = tf.Variable(tf.zeros([myLen]), dtype=tf.float32)
y = tf.reduce_sum(input_tensor=x_train*w)
loss = tf.abs(y-y_train)
optimizer = tf.compat.v1.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(loss)

# 初始化训练
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

#开始训练
loss_value = 0
for i in range(5000):
    myData = TrainData.getTrainData(myLen, w_true)
    i_print = i % 100 == 0
    had_iPrint = False
    if(i_print):
        print("Now is running Data NO.%d" %i)
        had_iPrint = True
    j = 0
    pre_loss = [0, 0]
    while(True):
        j = j+1
        loss_value = sess.run([x_train, w, loss, train, y], feed_dict={x_train: myData[0], y_train: myData[1]})
        if(j%100 == 0):
            if(not had_iPrint):
                print("Now is running Data NO.%d" %i)
                had_iPrint = True
            print("\tLoop of the Data is NO.%d: loss = %f" %(j, loss_value[2]))
        if(abs(loss_value[2] - pre_loss[0]) <= 0.000001):
            break
        pre_loss[0], pre_loss[1] = pre_loss[1], loss_value[2]
        

print("w = ")
print(loss_value[1])
print("w_true = ")
print(w_true)