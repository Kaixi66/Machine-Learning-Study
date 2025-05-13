# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

np.random.seed(24)
x = np.linspace(-10, 10, 100) # 成一个从 - 10 到 10（包含两端点）的等差数列，共 100 个元素。
# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

plt.plot(x, sigmoid(x)) # 将线性方程输出结果压缩在了0-1之间
#plt.show()

def logit_cla(yhat, thr=0.5):
    """
    逻辑回归类别输出函数：
    :param yhat: 模型输出结果
    :param thr：阈值
    :return ycla：类别判别结果
    """
    ycla = np.zeros_like(yhat)
    ycla[yhat >= thr] = 1
    return ycla

x = np.array([2, 0.5]).reshape(-1, 1)
yhat = sigmoid(1-x)    
# rint(logit_cla(yhat))