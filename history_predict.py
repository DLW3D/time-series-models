import os
import random
import tushare as ts
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

from evaluate_model import evaluate
from get_samples import get_samples, get_data, get_samples_targets


# 显示历史预测曲线
def history_predict(model, ts_code='600004.SH', date=20191128, delay=1, during=244, mod='simple'):
    # 获取数据
    df = get_data(ts_code)
    if df is None:
        print('can not find data')
        return
    # 打印历史准确率
    print(evaluate(model, ts_code=ts_code))
    # 整理数据
    data = get_samples(ts_code=ts_code, date=date, duiring=during)
    if data is None:
        return
    result = model.predict(data)
    print('数据分割日:', df[df['trade_date'].isin(['20180102'])].index[0])
    today = df[df['trade_date'].isin([date])].index[0] + 1

    # 画图
    if mod != 'complex' and mod != 'c':
        # 简单
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        axis_max = max(abs((df['close'].shift(-delay) - df['close'])[today - during:today]))
        plt.ylim(ymin=-axis_max, ymax=axis_max)
        ax1.plot((df['close'].shift(-delay) - df['close'])[today - during:today], c='b', label='目标时间后涨跌幅')
        ax1.set_ylabel('目标时间涨跌幅')
        ax2 = ax1.twinx()
        plt.ylim(ymin=0, ymax=1)
        ax2.plot(range(today - during, today), result * 0 + 0.5, c='r')
        ax2.plot(range(today - during, today), result * 0 + 0.7, c='r')
        ax2.plot(range(today - during, today), result * 0 + 0.9, c='r')
        ax2.plot(range(today - during, today), result, c='y', label='预测值')
        ax2.set_ylabel('预测值')
        # 图例
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        plt.title(ts_code)
        plt.show()
    else:
        # 复杂
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(df['close'][today - during:today], c='g', label='今天')
        ax1.plot(df['close'].shift(-delay)[today - during:today], c='b', label='目标时间')
        ax1.set_ylabel('走势')
        ax2 = ax1.twinx()
        plt.ylim(ymin=0, ymax=1)
        ax2.plot(range(today - during, today), result * 0 + 0.5, c='r')
        ax2.plot(range(today - during, today), result * 0 + 0.7, c='r')
        ax2.plot(range(today - during, today), result * 0 + 0.9, c='r')
        ax2.plot(range(today - during, today), result, c='y', label='预测值')
        ax2.set_ylabel('预测值')
        # 图例
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        plt.title(ts_code)
        plt.show()









