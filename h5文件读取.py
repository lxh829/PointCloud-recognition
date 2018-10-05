import h5py
f = h5py.File('lizi.h5')
data = f['data'][:]
label = f['label'][:]
print(label)
print(data)

