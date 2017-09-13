import h5py
import numpy as np

fname = "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
dfname = 'incpetion_owl.hdf5'

f = h5py.File(fname, 'r')
data_file = h5py.File(dfname, 'w')

for i in range(1, 95): #manually get from f.keys() 
    node_name = 'conv2d_' + str(i)
    conv_weight = f[node_name][node_name]['kernel:0'].value.tolist()
    data_file.create_dataset(node_name, data=conv_weight)


for i in range(1, 95): #manually get from f.keys()
    # Each BN layer has 3 members: beta, moving_mean, moving_variance 
    node_name = 'batch_normalization_' + str(i)
    bn_beta = f[node_name][node_name]['beta:0'].value.tolist()
    bn_mean = f[node_name][node_name]['moving_mean:0'].value.tolist()
    bn_var  = f[node_name][node_name]['moving_variance:0'].value.tolist()
    data_file.create_dataset(node_name + '_beta', data=bn_beta)
    data_file.create_dataset(node_name + '_mean', data=bn_mean)
    data_file.create_dataset(node_name + '_var',  data=bn_var)

node_name = 'predictions'
dense_weight = f[node_name][node_name]['kernel:0'].value.tolist()
dense_bias = f[node_name][node_name]['bias:0'].value.tolist()
data_file.create_dataset('linear_w', data=dense_weight) #(* just a random name *)
data_file.create_dataset('linear_b', data=dense_bias)

data_file.close()

# Read file 
# f = h5py.File(dfname)


## Inference: 
# import cv2, numpy as np 
# if __name__ == "__main__":
#    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#    im = np.expand_dims(im, axis=0)
#    model = keras.applications.VGG16()
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#    out = model.predict(im)
#    print np.argmax(out)