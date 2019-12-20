import os

import keras
import pandas as pd
import tushare as ts
from keras import backend as K


# 模型衡量标准
# 正样本中有多少被识别为正样本
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    real_true = K.sum(y_true)
    return true_positives / (real_true + K.epsilon())


def recall1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.2), 0, 1)))
    real_true = K.sum(y_true)
    return true_positives / (real_true + K.epsilon())


def recall2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.4), 0, 1)))
    real_true = K.sum(y_true)
    return true_positives / (real_true + K.epsilon())


# 负样本中有多少被识别为负样本
def n_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    real_true = K.sum(1 - y_true)
    return true_positives / (real_true + K.epsilon())


def n_recall1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * ((1 - y_pred) - 0.2), 0, 1)))
    real_true = K.sum(1 - y_true)
    return true_positives / (real_true + K.epsilon())


def n_recall2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * ((1 - y_pred) - 0.4), 0, 1)))
    real_true = K.sum(1 - y_true)
    return true_positives / (real_true + K.epsilon())


# 识别为正样本中有多少是正样本
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predict_true = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predict_true + K.epsilon())


def precision1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.2), 0, 1)))
    predict_true = K.sum(K.round(K.clip((y_pred - 0.2), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


def precision2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * (y_pred - 0.4), 0, 1)))
    predict_true = K.sum(K.round(K.clip((y_pred - 0.4), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


# 识别为负样本中有多少是负样本
def n_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    predict_true = K.sum(K.round(K.clip((1 - y_pred), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


def n_precision1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * ((1 - y_pred) - 0.2), 0, 1)))
    predict_true = K.sum(K.round(K.clip(((1 - y_pred) - 0.2), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


def n_precision2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip((1 - y_true) * ((1 - y_pred) - 0.4), 0, 1)))
    predict_true = K.sum(K.round(K.clip(((1 - y_pred) - 0.4), 0, 1)))
    return true_positives / (predict_true + K.epsilon())


# 预测结果中有多少是正样本
def prate(y_true, y_pred):
    return K.mean(K.round(K.clip(y_pred, 0, 1)))


# 实际中有多少是正样本
def trate(y_true, y_pred):
    return K.mean(K.round(K.clip(y_true, 0, 1)))


# 加载模型
def load_model(model_name='./model/cnn960to1080b.model', lookback=61, shape=5):
    # 加载模型时使用 keras.models.load_model(path, custom_objects=dependencies)
    dependencies = {
        'recall': recall,
        'recall1': recall1,
        'recall2': recall2,
        'precision': precision,
        'precision1': precision1,
        'precision2': precision2,
        'prate': prate,
        'trate': trate,
        'lookback': lookback,
        'shape': shape
    }

    model = keras.models.load_model(model_name, custom_objects=dependencies)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[recall, precision, recall1, precision1, recall2, precision2,
                           n_recall, n_precision, n_recall1, n_precision1, n_recall2, n_precision2, trate, prate])
    return model


# 获取一支股票的历史数据
def load_data(ts_code):
    # 判断文件是否存在,不存在则通过网络接口获得
    data_dir = '../data/'
    if not os.path.exists(data_dir + ts_code + '.csv'):
        # 初始化pro接口
        # pro = ts.pro_api('ed35503ddab56a5ae561f5fd7891d2b97b6546965aa98ba2df414bc4')
        # 获取前复权数据
        df = ts.pro_bar(ts_code=ts_code, adj='qfq')
        # 保存数据到文件
        if df is None:
            print('can not get data')
            return
        df.to_csv(data_dir + ts_code + '.csv', index=False)
    df = pd.read_csv(data_dir + ts_code + '.csv')
    # ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor
    # 股票代码, 交易日期, 开盘价, 最高价, 最低价, 收盘价, 昨收价, 涨跌额, 涨跌幅, 成交量, 成交额(千元)
    # 去空
    df.dropna(inplace=True)
    # 正序
    df = df.sort_index(ascending=False)
    # 索引重排序
    df.reset_index(drop=True, inplace=True)
    return df


# 加载股票列表
def load_code_list(market='SSE'):
    file_dir = '../data/' + 'code_list_' + market + '.csv'
    # 判断文件是否存在,不存在则通过网络接口获得
    if os.path.exists(file_dir):
        code_list = pd.read_csv(file_dir)
    else:
        # 初始化pro接口
        pro = ts.pro_api('ed35503ddab56a5ae561f5fd7891d2b97b6546965aa98ba2df414bc4')
        # 查询某交易所所有上市公司
        code_list = pro.stock_basic(exchange=market, list_status='L', fields='ts_code')  # ,symbol,name,market,list_date
        # 保存数据到文件
        code_list.to_csv(file_dir, index=False)
    code_list = code_list[['ts_code']].values.flatten()
    return code_list


# 根据模式输出
def print_verbose(verbose, text):
    if verbose:
        print(text)
