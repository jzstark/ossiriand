# Source: https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/predict-with-pretrained-model.ipynb

# Model: http://data.mxnet.io/mxnet/models/imagenet/inception-bn.tar.gz
# (http://data.mxnet.io/mxnet/models/imagenet/)

import mxnet as mx
import logging
import os
import numpy as np
from skimage import io, transform

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

model_dir = os.path.expanduser('~/Models/inception/4')

# Load the pre-trained model
# prefix = "Inception/Inception-BN"
prefix = os.path.join(model_dir, "Inception-BN")
num_round = 126
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

# mean image
mean_rgb = np.array([123.68, 116.779, 103.939])
mean_rgb = mean_rgb.reshape((3, 1, 1))

# if you like, you can plot the network
# mx.viz.plot_network(model.symbol, shape={"data" : (1, 3, 224, 224)})

# load synset (text label)
# synset = [l.strip() for l in open(os.path.join(model_dir, 'synset.txt')).readlines()]
synset = [l.strip() for l in open('Inception/synset.txt').readlines()]

def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean 
    normed_img = sample - mean_rgb
    print normed_img.shape, sample.shape, mean_rgb.shape
    normed_img = normed_img.reshape((1, 3, 224, 224))
    return normed_img

# Get preprocessed batch (single image batch)
batch = PreprocessImage(os.path.expanduser('~/Models/inception/panda.jpg'), True)
# Get prediction probability of 1000 classes from model
prob = model.predict(batch)[0]
# Argsort, get prediction index from largest prob to lowest
pred = np.argsort(prob)[::-1]
# Get top1 label
top1 = synset[pred[0]]
print("Top1: ", top1)
# Get top5 label
top5 = [synset[pred[i]] for i in range(5)]
print("Top5: ", top5)