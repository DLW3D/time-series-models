import numpy as np
from load_tools import *
from matplotlib import pyplot as plt


# 加载所需数据
def init(market_list, normalize=False):
    market_names = []
    market_datas = []
    market_datas_normal = []
    market_datas_date = []
    for i in range(len(market_list)):
        print('Load ', market_list[i])
        market_names.append([])
        market_datas.append([])
        market_datas_normal.append([])
        market_datas_date.append([])
        print('正在加载数据进入内存')
        for code_name in load_code_list(market=market_list[i]):
            print(code_name)
            market_names[i].append(code_name)
            market_datas[i].append(load_data(code_name))
        print('数据加载完毕')
        print('正在检查数据')
        for j in range(len(market_names[i])):
            # 查空
            if market_datas[i][j] is None:
                market_datas_normal[i].append(None)
                market_datas_date[i].append(None)
                continue
            if market_datas[i][j].empty:
                market_datas_normal[i].append(None)
                market_datas_date[i].append(None)
                continue
            # data = market_datas[i][j][['close', 'high', 'low', 'amount']].values
            data = market_datas[i][j][['close', 'open', 'high', 'low', 'amount']].values
            # 检查是否有错误值
            if np.isnan(data).any():
                print('nan in %s' % market_names[i][j])
                market_datas_normal[i].append(None)
                market_datas_date[i].append(None)
                continue
            # 进行正规化
            if normalize:
                mean = data.mean(axis=0)  # [6.98017146e+00, 7.12046020e+00, 6.83100609e+00, 1.65669341e+05]
                std = data.std(axis=0)  # [6.36818017e+00, 6.50689074e+00, 6.22204203e+00, 4.74562019e+05]
            else:
                mean = [0]
                std = [1]
            data -= mean
            if data.std(axis=0)[0] == 0:
                print('std is 0 in %s' % market_names[i][j])
                market_datas_normal[i].append(None)
                market_datas_date[i].append(None)
                continue
            data /= std
            market_datas_normal[i].append([data, mean, std])
            market_datas_date[i].append(market_datas[i][j]['trade_date'].tolist())
        print('数据检查完成')
    return market_names, market_datas, market_datas_normal, market_datas_date


# 加载所需数据
market_list = ['SSE', 'SZSE']
market_names, market_datas, market_datas_normal, market_datas_date = init(market_list, True)
date_list = market_datas[0][0]['trade_date'].values.tolist()


def get_data(ts_code):
    for i in range(len(market_list)):
        if ts_code in market_names[i]:
            return market_datas[i][market_names[i].index(ts_code)]


def get_data_normal(ts_code):
    for i in range(len(market_list)):
        if ts_code in market_names[i]:
            return market_datas_normal[i][market_names[i].index(ts_code)]


def get_data_date(ts_code):
    for i in range(len(market_list)):
        if ts_code in market_names[i]:
            return market_datas_date[i][market_names[i].index(ts_code)]


def get_code_list(market='SSE'):
    if market == 'ALL':
        ALL_names = []
        for i in range(len(market_list)):
            ALL_names += market_names[i]
        return ALL_names
    else:
        return market_names[market_list.index(market)]


# 训练历史可视化
def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    # plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')


# 显示训练过程
def plot_history(history):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    show_train_history(history, 'loss', 'val_loss')
    plt.subplot(2, 2, 2)
    show_train_history(history, 'recall', 'val_recall')
    plt.subplot(2, 2, 3)
    show_train_history(history, 'precision', 'val_precision')
    plt.subplot(2, 2, 4)
    show_train_history(history, 'precision2', 'val_precision2')
    plt.savefig('./model/auto_save.jpg')
    plt.show()


# 通过时间搜索index
def date2index(ts_code, start_date, end_date, lookback, delay, verbose=0):
    start = lookback - 1
    dl = get_data_date(ts_code)
    if not dl:
        return
    if start_date != '':
        if start_date not in dl:
            print_verbose(verbose, 'can not find date')
            return
        else:
            start = max(start, dl.index(start_date))
    end = len(dl) - delay
    if end_date != '':
        if end_date not in dl:
            print_verbose(verbose, 'can not find date')
            return
        else:
            end = min(end, dl.index(end_date))
    if start >= end:
        print_verbose(verbose, 'data range too small, may be date too close to boundary.')
        return
    return start, end
