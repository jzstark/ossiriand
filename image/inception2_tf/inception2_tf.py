# Source: https://github.com/tensorflow/models/blob/master/slim/slim_walkthrough.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import urllib2
import argparse
import numpy as np
import tensorflow as tf
from six.moves import urllib
import matplotlib.pyplot as plt

import inception_v1_net as inception
import inception_preprocessing
import inception_utils

slim = tf.contrib.slim
FLAGS = None
image_size = inception.inception_v1.default_image_size

def download_and_uncompress_tarball():
    """Downloads the `tarball_url` and uncompresses it locally.
    Args:
        tarball_url: The URL of a tarball file.
        dataset_dir: The directory where the temporary files are stored.
    """
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    tarball_url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
    dataset_dir = FLAGS.model_dir

    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if os.path.exists(filepath):
        return
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

def run_inference_on_image(image_url):
    """Runs inference on an image.
    Args:
      image: Image file name.
    Returns:
      Nothing
    """
    with tf.Graph().as_default():
        image_string = urllib2.urlopen(image_url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(FLAGS.model_dir, 'inception_v1.ckpt'),
            slim.get_model_variables('InceptionV1'))
        
        with tf.Session() as sess:
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
            
        plt.figure()
        plt.imshow(np_image.astype(np.uint8))
        plt.axis('off')
        plt.show()

    names = inception_utils.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

def main(_):
    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)

    download_and_uncompress_tarball()
    image = (FLAGS.image_file if FLAGS.image_file else
            os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
    run_inference_on_image(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.expanduser('~/Models/inception/2'),
        help="""\
        Path to .\
        """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg',
        help='Absolute path to image file. (only support url input now)'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)