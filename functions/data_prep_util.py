
####对数据的处理过程中用到的函数，将这些函数利用起来可以对数据进行处理，将ply文件保存成h5文件

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #功能得到此文件(data_prep_util.py)的绝对路径是
                                                       # C:\Users\Administrator\PycharmProjects\莫凡教程\pointnet-master\utils\data_prep_util.py
sys.path.append(BASE_DIR)  #把BASE_DIR得到的路径加入到系统路径当中
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty) #从plyfile文件中导入PlyData, PlyElement, make2d, PlyParseError, PlyProperty
import numpy as np
import h5py

SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')  #仅仅是连接BASE_DIR和后面的目录,并不会生成，若要生成使用此命令：os.mkdir(os.path.join("D:\","test.txt")),会生成D:\test.txt文件
 #采样路径                           #输出：C:/Users/Administrator/PycharmProjects/莫凡教程/pointnet-master/utils/third_party/mesh_sampling/build/pcsample

SAMPLING_POINT_NUM = 2048    #采样点数 2048
SAMPLING_LEAF_SIZE = 0.005   #采样叶子大小，差不多相当于学习率吧

MODELNET40_PATH = 'C:/Users/Administrator/PycharmProjects/莫凡教程/pointnet-master/data/modelnet40'
def export_ply(pc, filename): #定义 export_ply函数，pc  文件名
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])#np.zeros()全0填充，对于pc的位置0的维度，vertex(顶点)
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape  obj 形状的采样
def get_sampling_command(obj_filename, ply_filename):   #定义函数 ，变量为  obj文件名，ply文件名
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd
    #返回值为 cmd=SAMPLING_BIN + ' ' + obj_filename+' ' + ply_filename+SAMPLING_POINT_NUM+SAMPLING_LEAF_SIZE

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes 以下是加载modelnet40形状的帮助函数
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40    请阅读MODELNET40中的类别列表
def get_category_names():  #定义函数得到目录名称
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')  #路径拼接
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    print(shape_names)
    return shape_names

# Return all the filepaths for the shapes in MODELNET40   返回MODELNET40中所有形状的文件路径
def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
# 辅助函数创建父文件夹和所有子目录文件夹（如果不存在）
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files 以下是加载保存/加载HDF5文件的帮助函数
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename  将numpy数组数据和标签写入h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal,
		data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):  #正常写入h5文件，用这个
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename  将numpy数组数据和标签写入h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    data = []
    for j in data:
        data.append(j.encode())
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()




# Read numpy array data and label from h5_filename  从h5_filename读取numpy阵列数据和标签
def load_h5_data_label_normal(h5_filename):       #加载h5数据标签正常
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files  以下是加载保存/加载PLY文件的帮助函数
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))




if __name__=="__main__":
    # label = get_category_names()
    # get_obj_filenames()
    # points = read_ply('C:/Users/Administrator/PycharmProjects/莫凡教程/pointnet-master/utils/third_party/mesh_sampling/happy.ply')

    points = load_ply_data('C:/Users/Administrator/PycharmProjects/莫凡教程/pointnet-master/data/modelnet40/happy.ply',SAMPLING_POINT_NUM)
    label = get_category_names()
    # path = 'C:/Users/Administrator/PycharmProjects'
    file = save_h5("filelist.h5", points, label, data_dtype='uint8', label_dtype='uint8')

