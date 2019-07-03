import tensorflow as tf
from random import random

# 生成数据
myLen = 10

def getRandomList(length, multi):
    rList = []
    for i in range(length):
        rList.append(random()*multi)
    return rList

def getTrainData(w_true):
    data_x = getRandomList(myLen, 100)
    data_y = 0
    for i in range(myLen):
        data_y += w_true[i] * data_x[i]
    return [data_x, data_y]

w_true = getRandomList(myLen, 1)


# 已知的输入输出
x_train = tf.placeholder(shape=[myLen], dtype=tf.float32)
y_train = tf.placeholder(shape=[], dtype=tf.float32)

# 定义权重，预测输出函数，误差，训练方向
w = tf.Variable(tf.zeros([myLen]), dtype=tf.float32)
y = tf.reduce_sum(x_train*w)
loss = tf.abs(y-y_train)
optimizer = tf.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(loss)

# 初始化训练
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#开始训练
loss_value = 0
for i in range(500):
    myData = getTrainData(w_true)
    for j in range(500):
        loss_value = sess.run([x_train, w, loss, train, y], feed_dict={x_train: myData[0], y_train: myData[1]})
    print("i = %d, j = %d:" %(i, j))
    print("loss = %f" %(loss_value[2]))
    print("w = ")
    print(loss_value[1])

print("w_true = ")
print(w_true)