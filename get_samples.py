import random

import numpy as np

from get_tools import *


# 给出一支股票某段时间的Samples
def get_samples(ts_code='600004.SH', date=20191108, lookback=61, duiring=20, verbose=1, normalize=True):
    # 获取数据
    data_normal = get_data_normal(ts_code)
    if data_normal is None:
        print_verbose(verbose, 'can not find date normal')
        return
    # 获取标准化
    data = data_normal[0]
    mean = data_normal[1]
    std = data_normal[2]

    # 找到预测集
    se_index = date2index(ts_code, date, '', lookback, 0, verbose)
    if not se_index:
        print_verbose(verbose, 'can not get date')
        return
    i = se_index[0] + 1
    rows = np.arange(i - duiring + 1, i + 1)
    samples = np.zeros((len(rows),
                        lookback,
                        data.shape[-1]))
    # targets = np.zeros((len(rows),))
    for j, row in enumerate(rows):
        if rows[j] - lookback < 0:
            print_verbose(verbose, 'date range too small in %s' % ts_code)
            return
        indices = range(rows[j] - lookback, rows[j])
        samples[j] = data[indices]
    return samples


# 给出一支股票某段时间的Samples和Targets
def get_samples_targets(ts_code='600004.SH', start_date='', end_date='', lookback=61, delay=1, uprate=0.0, mod='', rand=False, verbose=1):
    # 获取数据
    data_normal = get_data_normal(ts_code)
    if data_normal is None:
        print_verbose(verbose, 'can not find date normal')
        return
    # 获取标准化
    data = data_normal[0]
    mean = data_normal[1]
    std = data_normal[2]

    # 找到起点终点位置
    se_index = date2index(ts_code, start_date, end_date, lookback, delay, verbose)  # 0.08
    if se_index is None:
        return
    start, end = se_index

    # 随机抽取一个
    if rand:
        start = random.randint(start, end - 1)
        end = start + 1

    # 构建
    rows = np.arange(start, end)
    samples = np.zeros((len(rows),
                        lookback,
                        data.shape[-1]))
    targets = np.zeros((len(rows),))
    for j, row in enumerate(rows):
        indices = range(rows[j] - (lookback - 1), rows[j] + 1)
        samples[j] = data[indices]
        # 涨跌值
        if mod == 'delta':
            targets[j] = (data[row + delay][0] * std[0] + mean[0]) - (data[row][0] * std[0] + mean[0])
            continue
        # 涨跌幅
        if mod == 'rate':
            targets[j] = (data[row + delay][0] * std[0] + mean[0]) / (data[row][0] * std[0] + mean[0]) - 1
            continue
        # 是否上涨
        if data[row + delay][0] * std[0] + mean[0] > (data[row][0] * std[0] + mean[0]) * (1 + uprate):
            targets[j] = 1
        else:
            targets[j] = 0
    return samples, targets


# 计算每只股票一段时间内的sample大小
def count_samples_weight(market, start_date='', end_date='', lookback=61, delay=1, verbose=1):
    code_list = get_code_list(market=market)
    names = []
    weight = []
    for code_name in code_list:
        print_verbose(verbose, code_name)
        df = get_data(code_name)
        if df is None:
            print_verbose(verbose, 'can not find data')
            continue
        # 找到起点终点位置
        se_index = date2index(code_name, start_date, end_date, lookback, delay, verbose)
        if se_index is None:
            continue
        start, end = se_index
        names.append(code_name)
        weight.append(end - start)
    return names, weight

