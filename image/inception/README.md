# InceptionV3 

Inception V3 model for Owl.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).


Re-implement from [Keras application](https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py), using pre-trained [weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)


# Reference
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
