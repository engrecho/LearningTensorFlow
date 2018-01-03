
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist2image
import numpy as np
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
BATCH = 100
# Mnist is Data
# sets(train=train, validation=validation, test=test)
# mnist.train.images, total 55000
# mnist.train.labels, total 55000
# mnist.validation.images, total 5000
# mnist.validation.labels, total 5000
# mnist.test.images, total 10000i
# mnist.test.labels, total 10000

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    # axis: A Tensor.
    # Must be one of the following types: int32, int64. int32,
    # 0 <= axis < rank(input).
    # Describes which axis of the input Tensor to reduce across.
    # 当axis=1时 返回每一行中的最大值的位置索引
    # 当axis=0时 返回每一列的最大值的位置索引

    correct_prediction = tf.equal(tf.argmax(prediction,axis = 1), tf.argmax(v_ys,axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(0.05).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 500 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

print("Learning Done \n")

INDEX = 21
your_num = input("Enter you Num:")
while( your_num != 'exit') :

    try:
        INDEX = int(your_num)
        image = np.reshape(mnist.test.images[INDEX,:],(1,28*28))
        predic_array   = sess.run(prediction,feed_dict={xs:image})
        predic_num     = np.argmax(predic_array,1)
        predic_percent = predic_array[0][predic_num]
        print('My Prediction is',predic_num,'and my confidence is',predic_percent*100,'%')
        mnist2image.mnist2image('test',INDEX).show_image()
    except Exception:
        print('Input Valid!!!!')

    your_num = input('Enter you number or press exit to end this:')

