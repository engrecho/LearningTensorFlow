import tensorflow as tf
import numpy as np

# M 假设是输入的图像矩阵，图像深度为1
M = np.array([
    [
        [[1],[2],[3]],
        [[4],[5],[6]],
        [[7],[8],[9]]
    ]
        ])
# M.shape = (1, 3, 3, 1)
# M[0,:,:,0] = [[1, 2, 3],
#               [5, 6, 7],
#               [8, 9, 0]]

# filter_weight，权重
# [2, 2, 1, 3] 代表 2*2过滤器大小，1代表输入的层数，3代表输出的层数
LEVEL = 3
W = np.array([
    [[[1,2,3]],
     [[4,5,6]]],
    [[[7,8,9]],
     [[0,1,2]]]
    ])

# W.shape = (3, 1, 2, 2)
# W[:,:,0,0] = [[1, 4],[7, 0]]
# W[:,:,0,1] = [[2, 5],[8, 1]]
# W[:,:,0,2] = [[3, 6],[9, 2]]



filter_weight = tf.get_variable('weights', [2, 2, 1, LEVEL], initializer = tf.constant_initializer(W))

biases = tf.get_variable('biases', [LEVEL], initializer = tf.constant_initializer(0))

M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
conv_with_bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(conv_with_bias, feed_dict={x: M})
    pooled_M = sess.run(pool, feed_dict={x: M})
    for i in range(3):
        print("convoluted_M",i,':\n', convoluted_M[0,:,:,i])

    print("pooled_M: \n", pooled_M)