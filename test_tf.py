import tensorflow as tf
import os

"""
	0 = all messages are logged (default behavior)
	1 = INFO messages are not printed
	2 = INFO and WARNING messages are not printed
	3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# with tf.device('/gpu:0'):
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    print (sess.run(c))