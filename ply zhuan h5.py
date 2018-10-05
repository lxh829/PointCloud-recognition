import numpy as np
import h5py
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)



def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    # data = []
    # for j in data:
    #     data.append(j.encode())
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


plydata = PlyData.read('PARTDESK.ply')
pc = plydata['vertex'].data[:1024]
pc_array_ = np.array([[x, y, z] for x, y, z in pc])
pc_array = np.expand_dims(pc_array_, 0)

print(pc_array)
print(type(pc_array))
label_ = [11]
label = np.expand_dims(label_, 0)

print(label)
print(type(label))

file = save_h5('PARTDESK.h5', pc_array, label, data_dtype='float', label_dtype='int')

f = h5py.File('PARTDESK.h5')
data = f['data']
label = f['label']
print(data)
print(type(data))

print(label)
print(type(label))