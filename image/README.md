# Usage

Suppose all models are stored at `~/Models`

Supported models:

## Classification

- [Inception1_tf.py](./incpetion1_tf.py):
    + Both graph and weights in one big .pb file. Key function: `tf.import_graph_def()`
    + Usage: `python inception1_tf.py --model_dir  --image_file --num_top_predictions`
- [Inception2_tf.py](./inception2_tf/inception2_tf.py):
    + Use `tf.contrib.slim`
    + Inception version 1 (can be updated ) . Only one .ckpt file. Create graph in a file.
    + Only support url image input now (can be updated ).
    + Usage: `python /inception2_tf/inception2_tf.py --model_dir --image_file url`
- [Inception_keras.py](./inception_keras/inception_keras.py):
    + Use Keras
    + Inception v3. Only one .h5 file. Create graph locally.
    + Usage: modify the image dir directly in the file and run without any parameters.
- [inception_mxnet.py](./inception_mxnet.py)
    + Use mxnet (compile and install); perquisite: `scikit-image`
    + Load graph and weight from files (in the format of .params and .json)
    + Usage: download model manually to ~/Models/inception/4; modify the dir of images directly in the script, and run. (Not ready to run yet)


## Image to Text

### Information

- name: [run_inference.py](./im2txt/run_inference.py)
- Original paper: [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge.](https://arxiv.org/abs/1609.06647)
- [model files repo](https://github.com/KranthiGV/Pretrained-Show-and-Tell-model)(License: MIT)
- Original repo: https://github.com/tensorflow/models/tree/master/im2txt
- License: Apache-2.0

## Description

- Image captioning: inference text description given an image of any size.
- Need to manually download the pre-trained model from the model files repo, which is trained on MOCOCO dataset.
- Pre-requisite: Do NOT need `nltk` package
- Usage: `python ./im2txt/run_inference.py --input_files picture1.jpg,picture2.jpg` (no space between picture names)
- Ouput: top-3 possible sentence that describes each given picture, together with possibility.

## Performance

- See the paper for performance test result.
- In the 2015 MS COCO challenge, this model ranked first position using both automatic and human evaluations.
