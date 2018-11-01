import h5py
import numpy as np

fname = "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
dfname = 'sqnet_owl.hdf5'

f = h5py.File(fname, 'r')
data_file = h5py.File(dfname, 'w')

count = 0
node_name = 'conv1'
conv_w = f[node_name][node_name+'_W:0'].value.tolist()
conv_b = f[node_name][node_name+'_b:0'].value.tolist()
data_file.create_dataset('conv2d_0:w', data=conv_w)
data_file.create_dataset('conv2d_0:b', data=conv_b)

count = 1
for i in range(2, 10):
    node_name = 'fire' + str(i)
    for layer in ['squeeze1x1', 'expand1x1', 'expand3x3']:
        conv_w = f[node_name][layer][node_name][layer + '_W:0'].value.tolist()
        conv_b = f[node_name][layer][node_name][layer + '_b:0'].value.tolist()
        data_file.create_dataset('conv2d_' + str(count) + ':w', data=conv_w)
        data_file.create_dataset('conv2d_' + str(count) + ':b', data=conv_b)
        count += 1

node_name = 'conv10'
conv_w = f[node_name][node_name+'_W:0'].value.tolist()
conv_b = f[node_name][node_name+'_b:0'].value.tolist()
data_file.create_dataset('conv2d_' + str(count) + ':w', data=conv_w)
data_file.create_dataset('conv2d_' + str(count) + ':b', data=conv_b)

print count

data_file.close()
