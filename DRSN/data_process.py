import os
from scipy.io import loadmat
import numpy as np
from collections import Counter
import scipy.io as sio

item_num = 100

def data_augment(path_0, person_num, fs, win_tlen, overlap_rate):
    """
        :param win_tlen: 滑动窗口的时间长度
        :param overlap_rate: 重叠部分比例, [0-100], 百分数:
                             overlap_rate*win_tlen*fs//100 是论文中的重叠量
        :param fs: 原始数据的采样频率
        :param data_iteror: 原始数据的生成器格式
        :return (X, y): X, 切分好的数据， y数据标签
                        X[0].shape == (win_len,)
                        X.shape == (N, win_len)
    """
    overlap_rate = int(overlap_rate)
    # 窗口的长度，单位采样点数
    win_len = int(fs * win_tlen)
    # 重合部分的时间长度，单位采样点数
    overlap_len = int(win_len * overlap_rate / 100)
    # 步长，单位采样点数
    step_len = int(win_len - overlap_len)

    # 滑窗采样增强数据
    X = []
    y = []

    for label in range(person_num):
        path = path_0 + str(label) + '/'
        X_ = []
        y_ = []
        for item in os.listdir(path):  #文件夹内的数据
            data_path = path + str(item)
            # 数据提取
            D = sio.loadmat(data_path)
            DD = D['phi_2']  ###
            ## DD是一个1*n的ndarray
            len_data = DD.size

            for start_ind, end_ind in zip(range(0, len_data - win_len, step_len),
                                      range(win_len, len_data, step_len)):
                X_.append(DD[0][start_ind:end_ind])
                y_.append(label)

            X.extend(X_[0:item_num])
            y.extend(y_[0:item_num])

    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess(path, person_num, fs, win_tlen,
               overlap_rate, random_seed):
    X, y = data_augment(path, person_num, fs, win_tlen, overlap_rate)
    # print(len(y[y==0]))

    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强后共有：{1}条,"
          .format(fs, X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
          .format(X.shape[1], int(overlap_rate * win_tlen * fs // 100)))
    print("-> 类别数据数目:", sorted(Counter(y).items()))
    return X, y


