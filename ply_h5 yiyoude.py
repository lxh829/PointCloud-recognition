import h5py

f = h5py.File('ply_data_test1.h5')
data = f['data'][22:23]
print(data)
label = f['label'][22:23]
print(label)



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



save_h5('car_0269.h5', data, label, data_dtype='float32', label_dtype='uint8')