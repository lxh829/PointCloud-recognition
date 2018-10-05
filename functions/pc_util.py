
######处理点云的实用函数,展示了一个实例

""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Draw point cloud
from eulerangles import euler2mat

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement

 
# ----------------------------------------
# Point Cloud/Volume Conversions   点云/体积转换
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    #将  点云数据  转换为 体积网格
    """ Input is BxNx3 batch of point cloud   b:batch批量 N:n个点  3：表示三维坐标，也是三通道
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    #将 体积网格  转换为  点云数据
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

#a = np.zeros((16,1024,3))
#print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

# ----------------------------------------
# Point cloud IO   点云IO   （使用过）
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format.  输入：Nx3，写入文件名为PLY格式。 """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers   简单点云和体积渲染器  （使用过）
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud   规范点云
    # We normalize scale to fit points in a unit sphere  我们规模化以适应单位球体中的点
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk   预先计算高斯盘
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer   通过z缓冲区订购点
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
    return image

def point_cloud_three_views(points):  #输入点云数据的三维数组  转变为  灰度图像  （使用过）
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """ 
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


from PIL import Image
def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function    演示draw_point_cloud功能 """
    points = read_ply('C:/Users/Administrator/PycharmProjects/莫凡教程/pointnet-master/utils/third_party/mesh_sampling/happy.ply')#调用了函数 read_ply()
    im_array = point_cloud_three_views(points)    #调用了 point_cloud_three_views()函数，在这个函数中调用了draw_point_cloud()函数
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('happy.jpg')

if __name__=="__main__":
    point_cloud_three_views_demo()


import matplotlib.pyplot as plt
def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array   点是一个Nx3的numpy数组 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)
