# 科学计算模块
import numpy as np
import pandas as pd
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.linear_model
# 自定义模块
from ML_basic_function import *

import sklearn
#sklearn默认接收的对象类型是数组，
# 即无论是特征矩阵还是标签数组，最好都先转化成array对象类型再进行输入。

# sklearn核心对象类型：评估器（estimator）
# 将评估器就理解成一个个机器学习模型，
# 而sklearn的建模过程最核心的步骤就是围绕着评估器进行模型的训练

np.random.seed(24)
features, labels = arrayGenReg(delta=0.01)

# sklearn中的线性回归评估器LinearRegression实际上是在sklearn包中的linear_model模块下
# 因为是类，实例化
model = sklearn.linear_model.LinearRegression()
# 需要输入上述数据对该模型进行训练。此时我们无需在输入的特征矩阵中加入一列全都是1的列：
X = features[:, :2]
y = labels

# 当fit方法执行完后，即完成了模型训练，此时model就相当于一个参数个数、参数取值确定的线性方程
model.fit(X, y)

# 查看自变量参数
print(model.coef_)
#查看模型截距
print(model.intercept_)

# 对比最小二乘法手动计算
print(np.linalg.lstsq(features, labels, rcond=-1)[0])

# 可以使用model中的predict方法进行预测
print("\n")
print(model.predict(X)[:10])
print("\n")
print(y[:10])

