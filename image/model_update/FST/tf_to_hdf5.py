import tensorflow as tf
import os
import h5py
import numpy as np

# The name of each layer

conv2d_n = 'conv2d'
conv_trans2d_n = 'transpose_conv2d'
norm_n = 'normalisation'
relu_n = 'activation'
add_n = 'add'

layer_names = [
    conv2d_n, norm_n, relu_n,
    conv2d_n, norm_n, relu_n,
    conv2d_n, norm_n, relu_n,
    conv2d_n, norm_n, relu_n, conv2d_n, norm_n, add_n,
    conv2d_n, norm_n, relu_n, conv2d_n, norm_n, add_n,
    conv2d_n, norm_n, relu_n, conv2d_n, norm_n, add_n,
    conv2d_n, norm_n, relu_n, conv2d_n, norm_n, add_n,
    conv2d_n, norm_n, relu_n, conv2d_n, norm_n, add_n,
    conv_trans2d_n, norm_n, relu_n,
    conv_trans2d_n, norm_n, relu_n,
    conv2d_n, norm_n
    ]

for i, val in enumerate(layer_names):
    layer_names[i] = val + '_' + str(i+1)
layer_names = [ x for x in layer_names if not x.startswith(relu_n)]

layer_names_rep = []
for val in layer_names:
    if val.startswith(norm_n):
        layer_names_rep.extend([val, val])
    else:
        layer_names_rep.append(val)

flag = True

for i, val in enumerate(layer_names_rep):
    if not val.startswith(norm_n):
        continue
    if flag == True:
        append = '_beta'; flag = False
    else:
        append = '_gamma'; flag = True
    layer_names_rep[i] = val + append

for l in layer_names_rep:
    if l.startswith(add_n):
        layer_names_rep.remove(l)


# Begin transfer
model_name = "wreck"
checkpoint_file = "checkpoint/" + model_name + ".ckpt"
reader = tf.train.NewCheckpointReader(checkpoint_file)

dfname = "fst_style_" + model_name + ".hdf5"
data_file = h5py.File(dfname, 'w')

# keys = reader.get_variable_to_shape_map().keys()
for i in range(len(layer_names_rep)):
    append = '' if i == 0 else '_' + str(i)
    key = 'Variable' +  append

    data_file.create_dataset(layer_names_rep[i], data=reader.get_tensor(key).tolist())
    print("tensor_name: ", key, " -- ", layer_names_rep[i])

data_file.close()
