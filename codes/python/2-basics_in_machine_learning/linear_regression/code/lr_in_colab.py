#@title Linear regression { display-mode: "both" }
# ex-2_1 linear regression
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
learning_rate = 1e-4 #@param {type:"number"}
num = 32 #@param {type: "integer"}
num_epoch = 50 #@param {type: "integer"}
sess = tf.Session()
 
x_input = tf.placeholder(tf.float32, shape=[None,], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None,], name='y_input')
w = tf.Variable(2.0, name='weight')
b = tf.Variable(1.0, name='biases')
y = tf.add(tf.multiply(x_input, w), b)
loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * num)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
 
'''tensorboard'''
gradients_node = tf.gradients(loss_op, w)
# print(gradients_node)
# tf.summary.scalar('norm_grads', gradients_node)
# tf.summary.histogram('norm_grads', gradients_node)
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('log')
 
init = tf.global_variables_initializer()
sess.run(init)
 
'''构造数据集'''
x_pure = np.random.randint(-10, 100, num)
x_train = x_pure + np.random.randn(num) / 10  # 为x加噪声
y_train = 3 * x_pure + 2 + np.random.randn(num) / 10  # 为y加噪声
Gradients = []
Loss = []
for i in range(num_epoch):
    _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
                                  feed_dict={x_input: x_train, y_input: y_train})
    print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))
    Gradients.append(gradients)
    Loss.append(loss)
fig = plt.figure(1, (14, 6))
AX = [fig.add_subplot(i) for i in range(121,123)]
name = ['Loss', 'Gradients']
color = ['r', 'b']
data = [Loss, Gradients]
for na, ax, co, d in zip(name, AX, color, data):
  ax.plot(np.linspace(0,num_epoch,num_epoch), d, co, label=na)
  ax.set_title(na)
  ax.legend()
 
sess.close()
