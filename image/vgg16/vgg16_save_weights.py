import h5py
import numpy as np

fname = "/home/stark/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
dfname = 'vgg16_owl.hdf5'

f = h5py.File(fname, 'r')
data_file = h5py.File(dfname, 'w')

# conv nodes
k = 1
for i in range(1, 6): # 5 blocks in total
    for j in range(1, 4):
        # This is how the author of keras network want to name each node
        if (j == 3 and (i == 1 or i == 2)): continue
        node_name_origin = 'block' + str(i) + '_conv' + str(j)
        conv_w = f[node_name_origin][node_name_origin + '_W_1:0'].value.tolist()
        conv_b = f[node_name_origin][node_name_origin + '_b_1:0'].value.tolist()

        node_name = 'conv2d_' + str(k)
        k += 1
        data_file.create_dataset(node_name + '_w', data=conv_w)
        data_file.create_dataset(node_name + '_b', data=conv_b)

assert(k == 14)

# fc nodes
for i in range(1, 3):
    node_name = 'fc' + str(i)
    fc_w = f[node_name][node_name + '_W_1:0'].value.tolist()
    fc_b = f[node_name][node_name + '_b_1:0'].value.tolist()
    data_file.create_dataset(node_name + '_w', data=fc_w)
    data_file.create_dataset(node_name + '_b', data=fc_b)

# prediction node
node_name = 'predictions'
p_w = f[node_name][node_name + '_W_1:0'].value.tolist()
p_b = f[node_name][node_name + '_b_1:0'].value.tolist()

data_file.create_dataset('fc3_w', data=p_w) # since the last node is also a fc node
data_file.create_dataset('fc3_b', data=p_b)

data_file.close()
f.close()

# Read file 
# f = h5py.File(dfname)