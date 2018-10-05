import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)


###得到合理的学习率
def get_learning_rate(batch):
    ###tf.train.exponential_decay（）函数作用 ：应用指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    ###tf.train.exponential_decay（learning_rate,global_step,decay_steps,decay_rate,staircase=False）
    # 函数作用 ：应用指数衰减的学习率。（最初学习率，用于衰减计算的全局步骤，衰退学习率的步数，衰退学习率，布尔值）
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


###训练开始
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):

            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE,
                                                                 NUM_POINT)  ###注①调用pointnet.py文件的placeholder_inputs()函数
            # pointclouds_pl的维数是：shape=(batch_size, num_point, 3)=（32,1024,3）
            # labels_pl的维数是：shape=(batch_size)=32

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize. 注意global_step = batch参数以最小化。
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            # 这会告诉优化器在每次训练时都会帮助您增加“批量”参数
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)  ###调用本文件85行定义的 get_bn_decay()函数
            tf.summary.scalar('bn_decay', bn_decay)  # 用来显示标量信息，在画loss,accuary时经常会用到这个函数

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl,
                                               bn_decay=bn_decay)  ###注②调用pointnet.py文件的get_model()函数，
            loss = MODEL.get_loss(pred, labels_pl, end_points)  ###注③调用pointnet.py文件的get_loss()函数
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            # tf.argmax(pred, 1)，返回的是pred中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
            # tf.to_int64(labels_pl)将labels_pl转化为64位整型
            # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，相反返回False，返回的值的矩阵维度和A是一样的
            # A = [[1,3,4,5,6]]  B = [[1,3,4,3,2]] --->[[ True  True  True False False]]

            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            # tf.cast(correct, tf.float32)将correct的bool类型的变量转换为float32的类型
            # tf.reduce_sum（）将转换过后的float32类型的correct，进行求和，然后除以BATCH_SIZE的大小（32）即为准确率
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)  ###调用本文件75行定义的 get_learning_rate()函数，得到合适的学习率
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.添加操作来保存和恢复所有变量
            saver = tf.train.Saver()

        # Create a session创建一个会话
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers添加摘要编写者
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        # log是事件文件所在的目录，这里是工程目录下的log目录。第二个参数是事件文件要记录的图，也就是tensorflow默认的图
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables  初始化变量
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()  # 这句代码的意思是刷新输出

            train_one_epoch(sess, ops, train_writer)  # 调用下面的train_one_epoch()函数
            eval_one_epoch(sess, ops, test_writer)  # 调用下面的eval_one_epoch()函数

            # Save the variables to disk.将变量保存到磁盘。
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    ###ops: dict mapping from string to tf ops 。 ops：dict从字符串映射到tf ops
    is_training = True

    # Shuffle train files  随机播放训练文件
    train_file_idxs = np.arange(0, len(TRAIN_FILES))  # np.arange(起点，终点，步长)步长默认为1，0,1,2,3,4
    np.random.shuffle(train_file_idxs)  # 随机排列

    ###训练数据的读入
    # 训练文件一共有5个，0-4，fn分别为0,1,2,3,4，开始循环
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        # 调用provider.py文件的loadDataFile()函数和load_h5()函数，从H5文件中读取正确的数据对应的正确的标签
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]  # 应该是对数据进行了某些变换，要不要都行应该
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(
            current_label))  # 调用provider.py中shuffle_data()函数
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]  # file_size是2048
        num_batches = file_size // BATCH_SIZE  # BATCH_SIZE=32,2048/32=64，num_batches=64

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):  # (batch_idx是从0-63，num_batches=64)
            start_idx = batch_idx * BATCH_SIZE  # 0,64  64,128  128，256  一直类推
            end_idx = (batch_idx + 1) * BATCH_SIZE  # 第一轮 start_idx=0,end_idx=64

            # 通过旋转和抖动增加批量点云
            rotated_data = provider.rotate_point_cloud(
                current_data[start_idx:end_idx, :, :])  # 调用provider.py中的rotate_point_cloud()函数
            jittered_data = provider.jitter_point_cloud(rotated_data)

            # 输入处理过后的点云数据，将此数据输送到前面169行的ops中，开开始训练
            feed_dict = {ops['pointclouds_pl']: jittered_data,  # 点云数据
                         ops['labels_pl']: current_label[start_idx:end_idx],  # 点云数据对应的正确标签，一共分为64组，一组32个，仅仅是针对1个训练的h5文件
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            # 下面几行是求平均损失和正确率
            pred_val = np.argmax(pred_val, 1)  # tf.argmax(pred_val, 1)，返回的是pred_val中的最大值的索引号
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct  # total_correct = total_correct + correct,累加求和
            total_seen += BATCH_SIZE  # total_seen = total_seen + BATCH_SIZE
            loss_sum += loss_val  # loss_sum = loss_sum + loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


# 一些代码和上面的train_one_epoch()函数一致，只不过是换了数据集
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """  # ops：dict从字符串映射到tf ops
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
