# -*- coding: utf-8 -*-

# Source: https://github.com/buriburisuri/speech-to-text-wavenet

import sugartensor as tf
import numpy as np
import librosa
import os
from model import *
#import data

model_dir = os.path.expanduser('~/Models/sttw/')

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size

#
# inputs
#

# vocabulary size
index2byte = ['<EMP>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

voca_size = len(index2byte)

# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_

# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))


# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))
# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)
# encode audio feature
logit = get_logit(x, voca_size=voca_size)
# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)
# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
#
# regcognize wave file
#
# command line argument for input wave file path
tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# load wave file
wav, _ = librosa.load(tf.sg_arg().file, mono=True, sr=16000)
# get mfcc feature
mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    # run session
    label = sess.run(y, feed_dict={x: mfcc})

    # print label
    print_index(label)