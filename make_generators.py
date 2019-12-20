import os
import random

import tushare as ts
import numpy as np
import pandas as pd

from get_tools import *
from get_samples import count_samples_weight


def make_generators(ts_code='600004.SH', batch_size=1024, start_rate='', end_rate='', shuffle=True):
    # 获取数据
    df = get_data(ts_code)
    if df is None:
        return
    # 筛选出 收盘价, 最高价, 最低价, 成交量
    df = df[['close', 'high', 'low', 'amount']]
    data = df.values
    # 计算错误行
    # errindex = []
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         if data[i, j] == 0 or np.isnan(data[i, j]):
    #             errindex.append(i)
    # data = np.delete(data, errindex, axis=0)  # 删除错误行
    # 检查是否有错误值
    if np.isnan(data).any():
        print('nan in %s' % ts_code)
        return

    # 定义参数
    lookback = 261  # 观察将追溯261个交易日(大概一年)。
    step = 1  # 观测将在每天一个数据点取样。
    delay = 22  # 目标是未来第22个交易日(大概一月)。
    uprate = 0.0  # 预测目标时间是否上涨

    # 定义分割数据集
    row = len(data)
    if start_rate == '':
        start = 0
    else:
        start = int(np.floor(start_rate * row))
    if end_rate == '':
        end = row
    else:
        end = int(np.floor(end_rate * row))

    weight = end - start - lookback - delay
    if batch_size == 'auto':
        batch_size = weight
    # 检查数据大小是否够一个batch_size
    if weight // batch_size < 1:
        print('%s(%s) is too small for a batch_size(%s)' % (ts_code, weight, batch_size))
        return

    # 标准化
    mean = data.mean(axis=0)  # [6.98017146e+00, 7.12046020e+00, 6.83100609e+00, 1.65669341e+05]  #     # [0]
    data -= mean
    std = data.std(axis=0)  # [6.36818017e+00, 6.50689074e+00, 6.22204203e+00, 4.74562019e+05]  #   #[1]
    data /= std

    # 数据生成器
    def generator(data, lookback, delay, uprate, min_index, max_index,
                  shuffle=True, batch_size=128, step=1):
        """
        :param data: 数据
        :param lookback: 判断依据回溯时间
        :param delay: 预测目标延迟时间
        :param uprate: 预测目标提升比例
        :param min_index: 使用的数据开始位置
        :param max_index: 使用的数据结束位置
        :param shuffle: 是否打乱
        :param batch_size: 批大小
        :param step: 数据中取数据的间隔
        :return: X[batch_size, None, data.shape[1]], y[batch_size,]
        """
        if max_index is None:
            max_index = len(data) - delay
        else:
            max_index -= delay
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                if data[rows[j] + delay - 1][0] * std[0] + mean[0] > (samples[j][-1][0] * std[0] + mean[0]) * (
                        1 + uprate):
                    targets[j] = 1
                else:
                    targets[j] = 0
            yield samples, targets

    # 构建生成器
    train_gen = generator(data,
                          lookback=lookback,
                          delay=delay,
                          uprate=uprate,
                          min_index=start,
                          max_index=end,
                          shuffle=shuffle,
                          step=step,
                          batch_size=batch_size)
    return train_gen, weight











