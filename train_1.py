###对train.py中函数train_one_epoch（）的解释以及每一步输出的结果
import h5py
import provider
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))


def loadDataFile(filename):
    print(filename)
    return load_h5(filename)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


train_file_idxs = np.arange(0, len(TRAIN_FILES))

if __name__ == "__main__":

    current_data0, current_label0 = loadDataFile(TRAIN_FILES[train_file_idxs[0]])
    print(current_data0)
    print(current_label0)

    current_data1 = current_data0[:, 0:1024, :]
    print(current_data1)

    current_data2, current_label1, _ = provider.shuffle_data(current_data1, np.squeeze(current_label0))
    print(current_data2)
    print(current_label1)

    current_label2 = np.squeeze(current_label1)
    print(current_label2)

    file_size = current_data1.shape[0]
    print(file_size)

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(64):

        start_idx = batch_idx * 64
        end_idx = (batch_idx + 1) * 64
        print(start_idx)
        print(end_idx)

        rotated_data = provider.rotate_point_cloud(current_data2[start_idx:end_idx, :, :])
        print(rotated_data)

        jittered_data = provider.jitter_point_cloud(rotated_data)
        print(jittered_data)



