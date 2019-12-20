import os
import shutil
import time

# 重置data文件夹
data_dir = '../data/'
if os.path.exists(data_dir+"code_list_SSE.csv"):
    shutil.rmtree(data_dir)
    os.mkdir(data_dir)

from serch_predict import *
from evaluate_model import *
from history_predict import *

# 加载模型
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
model = load_model(model_name='./model/binary/ATT140to740.model')
# model = load_model(model_name='./model/1y1d/cudnnGRU/cudnnGRU210to340conv.model')

# 开始搜索
today = time.strftime("%Y%m%d")
print('today : %s' % today)
result_code, result_pred, rate_pred = search_predict(model, date=today, duiring=1, market='ALL', verbose=1, baseline=0.9)
# 将结果保存到文件
f = open("result.txt", "w")
f.write('rate_pred:\n%s\n' % rate_pred)
for i in range(len(result_code)):
    f.write('%s*****************************************\n%s\n' % (result_code[i], result_pred[i]))
f.close()

