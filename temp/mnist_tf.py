from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_", y)

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    saver.save(sess, 'my-model')

def meta2txt():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('my-model.meta')
        foo = new_saver.export_meta_graph()
        with open('graph.prototxt', 'r+w') as f:
            f.write(str(foo))


def inference():

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('my-model.meta')
        new_saver.restore(sess, './my-model')

        # How to get all possible keys in the current collection?

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_ = tf.get_collection("y_")[0]

        print(y, y_)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              y_: mnist.test.labels}))

def retrain():
    pass


def main(_):
    # Import data
    # train()
    # meta2txt()
    inference()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# Two disadvanges here:
# First, the original programmer has to add all possible variables to the collection
# Second, the user has to use these variables explicitly, knowing all possible names in the original 