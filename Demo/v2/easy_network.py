#! /usr/bin/python3
import sys
sys.path.append("./function")

import tensorflow as tf
import TrainData

# disable eager in tensorflow v2
tf.compat.v1.disable_eager_execution()

# 定义数据
myLen = 500
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
for i in range(1000):
    myData = TrainData.getTrainData(myLen, w_true)
    for j in range(50000):
        loss_value = sess.run([x_train, w, loss, train, y], feed_dict={x_train: myData[0], y_train: myData[1]})
    print("i = %d:" %i)
    print("\tloss = %f" %(loss_value[2]))
    print("\tw = ", loss_value[1])

print("w_true = ")
print(w_true)