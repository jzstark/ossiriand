# Usage

Suppose all models are stored at `~/Models` 

Supported models:

- [Inception1_tf.py](./incpetion1_tf.py): 
    + Both graph and weights in one big .pb file. Key function: `tf.import_graph_def()`
    + Usage: `python inception1_tf.py --model_dir  --image_file --num_top_predictions`
- [Inception2_tf.py](./incpetion2_tf/incpetion2_tf.py): 
    + Inception version 1 (can be updated ) . Only one ckpt file. Create graph in a file. 
    + Only support url image input now (can be updated ). 
    + Usage: `python /incpetion2_tf/incpetion2_tf.py --model_dir --image_file url`