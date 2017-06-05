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


