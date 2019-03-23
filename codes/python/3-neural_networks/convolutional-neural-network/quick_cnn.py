#@title Quick_CNN { display-mode: "both" }
# # coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import reduce

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	# initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(pre_layer, depth):
	pre_layer_shape = pre_layer.get_shape().as_list()
	layer_shape = [3, 3, pre_layer_shape[-1], depth]
	w = tf.Variable(tf.truncated_normal(layer_shape, stddev=0.1))
	b = tf.constant(0.1, shape=[depth])
	h_conv = tf.nn.relu(tf.nn.conv2d(pre_layer, w, strides=[1, 1, 1, 1], padding='SAME') + b)
	h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return h_pool

if __name__ == '__main__':

	mnist = input_data.read_data_sets('/content/GoogleDrive/Python27/MNIST_data', one_hot=True)
	batch_size = 32
	
	with tf.name_scope('Input'):
		x = tf.placeholder("float", shape=[None, 784])
		y_ = tf.placeholder("float", shape=[None, 10])
		x_image = tf.reshape(x, [-1,28,28,1])
		# x_image_sum = tf.summary.image('input_images', x_image)
	
	# ------------------conv-----------------------------------
	layer = x_image
	for layer_i in range(4):
		layer = conv_layer(layer, 64)
	# --------------fc--------------------------------------
	layer_shape = layer.get_shape().as_list()
	num_f = reduce(lambda a,b:a * b, layer_shape[1:])
	# num_f = layer_shape[]
	W_fc1 = weight_variable([num_f, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(layer, [-1, num_f])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	
	with tf.name_scope('Loss'):
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	with tf.name_scope('Train'):
		train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
	with tf.name_scope('Accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		accuracy_sum = tf.summary.scalar('accuracy', accuracy)
	
	sess = tf.Session()
	# sess.run(tf.initialize_all_variables())
	# writer = tf.summary.FileWriter("E:\Anaconda2\Programs\Tensorboard", sess.graph)
	sess.run(tf.global_variables_initializer())
	# merged = tf.summary.merge([x_image_sum, w_conv1_sum, loss_sum, accuracy_sum, conv1_output])
	
	# saver = tf.train.Saver(max_to_keep=1) # 定义保存3个模型
	# max_acc = 0
	
	for i in range(100):
		batch = mnist.train.next_batch(batch_size)
		# sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		_, acc, loss = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		# writer.add_summary(rs, i)
		step = i+1
		print("When the cross_entropy is %.2f, accuracy on training data at step %s is %.2f ." %(loss, step, acc))
		if i%10 == 0:
			print('\n')
			test_accuracy = sess.run(accuracy, feed_dict={
									x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
			print('Accuracy on testing data is %.2f .' %(test_accuracy))
		# sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
		# if (acc > max_acc) & (i > 399): # 保存精度高的三个模型
			# max_acc = acc
			# saver.save(sess, r'E:\Anaconda2\Programs\Tensorboard\f_map.ckpt', global_step=i+1)
		
	# test_image, test_label = mnist.test.images[100,:].reshape((1,-1)), mnist.test.labels[0,:].reshape((1,-1))
	# features1 = sess.run(h_pool1, feed_dict={x: test_image, y_: test_label, keep_prob: 1.0})
	
	# print("test accuracy %g"%sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	
	sess.close()
    
