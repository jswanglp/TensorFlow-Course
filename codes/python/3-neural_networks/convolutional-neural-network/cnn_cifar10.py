#@title datasets_tutorials(cifar-10) { display-mode: "both" }
# conding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from functools import reduce
import tensorflow_datasets as tfds
import numpy as np
import time

def format_tran(tfdata, batch_size=32):
    batch_tfdata = tfdata.shuffle(1).batch(batch_size)
    batch_imgs = tfds.as_numpy(batch_tfdata).__next__()['image']
    batch_labels = tfds.as_numpy(batch_tfdata).__next__()['label']
    return batch_imgs, batch_labels

# tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':
    # filepath = '/content/GoogleDrive/Python27/MNIST_data'
    # # filepath = r'E:\Anaconda2\Programs\MNIST_data'
    # mnist = input_data.read_data_sets(filepath, one_hot=True)
    # mnist_train = tfds.load("mnist", split=tfds.Split.TRAIN)
    mnist_train = tfds.as_numpy(tfds.load("cifar10", split=tfds.Split.TRAIN, batch_size=-1))
    imgs_train, labels_train = mnist_train['image'].reshape(-1, 3072) / 255., mnist_train['label']
    # imgs_train, labels_train = tf.reshape(mnist_train['image'], shape=[-1, 784]), tf.one_hot(mnist_train['label'], depth=10)

    mnist_test = tfds.as_numpy(tfds.load("cifar10", split=tfds.Split.TEST, batch_size=-1))
    # mnist_test = tfds.load("mnist", split=tfds.Split.TEST, batch_size=-1)
    imgs_test, labels_test = mnist_test['image'].reshape(-1, 3072) / 255., mnist_test['label']

    learning_rate = 3e-4 #@param {type:"number"}
    batch_size = 256 #@param {type:"integer"}
    num_epochs = 80 #@param {type:"integer"}

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 3072])
        y_p = tf.placeholder(tf.int64, shape=[None, ])
        y = tf.one_hot(y_p, depth=10)
        keep_pro = tf.placeholder(tf.float32)

        x_imgs = tf.reshape(x, shape=[-1, 32, 32, 3], name='input_images')
        w_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1), name='weights_conv1')
        b_1 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_conv1')
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_imgs, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        w_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='weights_conv2')
        b_2 = tf.Variable(tf.constant(0.1, shape=[128]), name='bias_conv2')
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # layer_shape = h_pool2.get_shape().as_list()
        # num_f = reduce(lambda a,b:a * b, layer_shape[1:])
        # h_pool2_fla = tf.reshape(h_pool2, shape=[-1, num_f])
        h_pool2_fla = tf.layers.flatten(h_pool2)
        num_f = h_pool2_fla.get_shape().as_list()[-1]
        
        w_fc1 = tf.Variable(tf.truncated_normal([num_f, 256], stddev=0.1), name='weights_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[256]), name='bias_fc1')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_fla, w_fc1) + b_fc1)
        h_drop1 = tf.nn.dropout(h_fc1, keep_prob=keep_pro, name='Dropout')

        w_fc2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), name='weights_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='bias_fc2')
        h_fc2 = tf.matmul(h_drop1, w_fc2) + b_fc2
        
        # tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_fc1)
        # regularizer = tf.contrib.layers.l2_regularizer(scale=1500./60000)
        # reg_tem = tf.contrib.layers.apply_regularization(regularizer)

        with tf.name_scope('loss'):
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc2))
            # entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc2) + reg_tem)
        
        with tf.name_scope('accuracy'):
            prediction = tf.cast(tf.equal(tf.arg_max(h_fc2, 1), tf.argmax(y, 1)), "float")
            accuracy = tf.reduce_mean(prediction)
        
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(entropy_loss)

        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            # batch_imgs, batch_labels = format_tran(mnist_train, batch_size=batch_size)

            for num in range(num_epochs):
                # batch = mnist.train.next_batch(batch_size)
                # batch_imgs, batch_labels = format_tran(mnist_train, batch_size=batch_size)
                # imgs_train, labels_train = batch_imgs.reshape(-1, 784), batch_labels
                imgs_data = np.c_[imgs_train, labels_train]
                np.random.shuffle(imgs_data)
                num_batchs = imgs_train.shape[0] // batch_size
                start = time.time()
                for num_ep in range(num_batchs):
                    # start = time.time()
                    imgs_batch = imgs_data[num_ep*batch_size:(num_ep+1)*batch_size, :-1]
                    labels_batch = imgs_data[num_ep*batch_size:(num_ep+1)*batch_size,-1]
                    _, acc, loss = sess.run([train_op, accuracy, entropy_loss], feed_dict={x: imgs_batch,
                                                                                        y_p: labels_batch,
                                                                                        keep_pro: 0.5})
                end = time.time()
                acc *= 100
                num_e = str(num + 1)
                print_list = [num_e, loss, acc]
                print("Epoch {0[0]}, train_loss is {0[1]:.4f}, accuracy is {0[2]:.2f}%.".format(print_list))
                print("Running time is {0:.2f}s.".format(end-start))
            _, acc, loss = sess.run([train_op, accuracy, entropy_loss], feed_dict={x: imgs_test,
                                                                                    y_p: labels_test,
                                                                                    keep_pro: 1.})
            acc *= 100
            print_list = [loss, acc]
            print("Test_loss is {0[0]:.4f}, accuracy is {0[1]:.2f}%.".format(print_list))
        
        sess.close()