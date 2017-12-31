
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # Y

# FOR matplotlib ,please install on command line
# python -mpip install -U pip
# python -mpip install -U matplotlib

def HiddenLayer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal(shape=(in_size,out_size)))
    biases  = tf.Variable(tf.zeros(shape=(1, out_size)))
    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data

row_size = 500
col_size = 1
NoisePw  = 10


x_data  = (2*np.random.rand(row_size,col_size)-1)*10
x_data  = x_data[np.argsort(x_data[:,0])]  #x_data has been sorded for plotting, DO NOT use x_data.sort()
noise   = np.random.normal(0, NoisePw , x_data.shape)
y_data  = 2.1* np.square(x_data) + 14 + noise

fig    = plt.figure(1)
ax     = fig.add_subplot(2,1,1)
ax_los = fig.add_subplot(2,1,2)


if 1:
    ax.plot(x_data,y_data)
    plt.ion()
    plt.show()
    print(x_data)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,shape=(None,1))
ys = tf.placeholder(tf.float32,shape=(None,1))


# add hidden layer and output layer
HiddenNode  = HiddenLayer(xs, 1, 10 , activation_function=tf.sigmoid)
prediction  = HiddenLayer(HiddenNode , 10, 1, activation_function=None)

# the loss between prediction and real data
global_step = tf.Variable(0)
# use exponential_decay
# learning_rate = init_learning_rate * decay_rate ^(global_step/decay_steps),
learning_rate = tf.train.exponential_decay(0.2,global_step,50,0.95,staircase=True)
# use MSE , Don't use Cross_Entropy
loss = tf.reduce_mean(tf.square(y_data - prediction))
# use AdamOptimizer instead of GradientDescentOptimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


feed = {xs: x_data, ys: y_data}
for i in range(10000):
    # training
    sess.run(train_step, feed_dict=feed)
    if i % 100 == 60:
        # to see the step improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        print('after',i,'turn,loss is:',sess.run(loss, feed_dict=feed))
        lines = ax.plot(x_data, sess.run(prediction, feed_dict=feed))
        ax_los.plot(i,sess.run(loss, feed_dict=feed),color='r',marker = '.',linewidth=0.1)
        plt.pause(0.1)

plt.ioff()
plt.show()
