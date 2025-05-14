# 科学计算模块
import numpy as np
import pandas as pd
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
# 自定义模块
from ML_basic_function import *

np.random.seed(24)
# step1 拿到特征矩阵（x）和标签数组（y）
features, labels = arrayGenReg(delta=0.01)
# step2 模型选取，此处我们选取带有bias的多元线性方程进行建模
# W = [w1, w2, b] features = [x1, x2, 1]

#step3 构建损失函数，我们先构建一组人工W来计算SSE(这里的w也就是函数的系数)
np.random.seed(24)
w = np.random.randn(3).reshape(-1, 1)# 取三个W

# 构造损失函数
def SSELoss(X, w, y):
    """
    SSE计算函数
    
    :param X：输入数据的特征矩阵
    :param w：线性方程参数
    :param y：输入数据的标签数组
    :return SSE：返回对应数据集预测结果和真实结果的误差平方和 
    """
    y_hat = X.dot(w)
    SSE = (y - y_hat).T.dot(y - y_hat)
    return SSE
# 简单测试
# print(SSELoss(features, w, labels))

# step4 利用最小二乘法求解损失函数，找到最佳参数取值，使得模型预测结果和真实值接近
# 利用公式：w = inverse(X.T * X) * X.T * y 
# 在求解过程中需要特征矩阵（x）的交叉乘积可逆(X.T * X),通过行列式可以判断是否满足
# print(np.linalg.det(features.T.dot(features)))
# 行列式不为0，则存在逆矩阵

#基础方法求解
w = np.linalg.inv(features.T.dot(features)).dot(features.T).dot(labels)
# print(SSELoss(features, w, labels)) # 测试模型SSE指标

# lstsq求解，利用此函数求解最小二乘法，结果一致
#一共有4个返回值，只用取第一个求解结果就行
#print(np.linalg.lstsq(features, labels, rcond=-1)[0]) #用于处理矩阵奇异值的阈值（默认 -1，表示使用机器精度）

'''
最小二乘法的限制：
1. 非线性关系且存在一定的白噪声时受影响较大
2. 如果特征矩阵(X)的交叉乘积不可逆则不能使用，此时代表着严重的共线性
代表着特征矩阵有线性关系，可以从三个方面下手：
1.对数据进行降维处理，在SVD过程中会对数据进行正交变换，使得数据线性无关
2.修改损失函数方法，求解广义逆矩阵得到近似最优解。此外还可以使用梯度下降来进行求解
3.修改损失函数，如果XTX不可逆，则可以通过加入一个正则化项
'''

'''
对于线性回归模型，我们还可以使用决定系数(R-square,拟合优度检验)作为模型评估指标。
SSR由预测数据和标签均值之间差的平方和计算。SST由实际值域均值之间的差的平方和来计算。
决定系数 = SSR/SST = 1-SSE/SST
是一个[0, 1]之间的值,月接近1效果越好。
'''
sst = np.power(labels - labels.mean(), 2).sum()
sse = SSELoss(features, w, labels)
r = 1-(sse/sst)
print(r)