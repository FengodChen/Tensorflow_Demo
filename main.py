import tensorflow as tf

x_train = tf.placeholder(shape=[3], dtype=tf.float32)
y_train = tf.placeholder(shape=[], dtype=tf.float32)

w = tf.Variable(tf.zeros([3]), dtype=tf.float32)

y = tf.reduce_sum(x_train*w)
loss = tf.abs(y-y_train)

optimizer = tf.train.RMSPropOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(50000):
  loss_value = sess.run([x_train, w, loss, train, y], feed_dict={x_train:[1, 2, 3], y_train: 2})
  if(i % 1000 == 0):
    print(loss_value)
