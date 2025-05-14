# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *


# Step1 确定数据及其模型
# step2 设置初始参数值

np.random.seed(24)
w = np.random.randn(2, 1) # 参数默认列向量

#Step3 根据损失表达式求出梯度表达式
def MSELoss(X, w, y):
    """
    MSE指标计算函数
    """
    SSE = SSELoss(X, w, y)
    MSE = SSE / X.shape[0]
    return MSE   

features = np.array([1, 3]).reshape(-1, 1)
features = np.concatenate((features, np.ones_like(features)), axis=1)
labels = np.array([2, 4]).reshape(-1, 1)

# 此处我们使用MSE作为损失函数，对应的梯度表达式为
def lr_gd(X, w, y):
    """
    线性回归梯度计算公式
    """
    m = X.shape[0]
    grad = 2 * X.T.dot((X.dot(w) - y)) / m
    return grad

#Step4 执行梯度下降，weight -= lr * lr_gd()
def w_cal(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    """
    梯度下降中参数更新函数 
    :param X: 训练数据特征
    :param w: 初始参数取值
    :param y: 训练数据标签
    :param gd_cal：梯度计算公式
    :param lr: 学习率
    :param itera_times: 迭代次数       
    :return w：最终参数计算结果   
    """
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
    return w

np.random.seed(24)
w = np.random.randn(2, 1)
w = w_cal(features, w, labels, gd_cal = lr_gd, lr = 0.1, itera_times = 100)
print(w)
print(SSELoss(features, w, labels))