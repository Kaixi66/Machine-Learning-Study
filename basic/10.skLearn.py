# 科学计算模块
import numpy as np
import pandas as pd
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics
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

# 在metrics模块下导入MSE计算函数
test = sklearn.metrics.mean_squared_error(model.predict(X), y)
print(test)


# estimator中Parameters部分就是当前模型超参数的相关说明
model1 = sklearn.linear_model.LinearRegression(fit_intercept=False) 
#不带截距的线性方程组
print('\n', model1.get_params())

# 超参数可以更改
model1.set_params(fit_intercept=True)

model.set_params(fit_intercept=False)
# 还没给x和y时，模型参数还不会进行修改
print(model.coef_, model.intercept_) #还带有截距
model.fit(X, y)
print("\n", model.coef_, model.intercept_)


# sklearn创建数据集
# 鸢尾花数据
from sklearn.datasets import load_iris
iris_data = load_iris()
print(iris_data)
print("\n", type(iris_data)) # bunch类型，类似字典类型的对象
#data为特征矩阵，target为标签
print(iris_data.data[:10])
print("\n", iris_data.target[:10])

# 返回各列名称
print("\n", iris_data.feature_names)

#返回标签各类名称
print("\n", iris_data.target_names[:10])

# 只返回特征矩阵和标签数组这两个对象
X, y = load_iris(return_X_y=True)
print(X[:10])
print(y[:10])

# 创建dataframe对象
iris_dataFrame = load_iris(as_frame=True)
print(iris_dataFrame.frame)

# sklearn中数据集切分方法
from sklearn.model_selection import train_test_split

# sklearn中的数据标准化与归一化
# 从功能上划分，sklearn中的归一化其实是分为标准化（Standardization）
# 和归一化（Normalization）两类
# 其中，此前所介绍的Z-Score标准化和0-1标准化，都属于Standardization的范畴
# Normalization则特指针对单个样本（一行数据）利用其范数进行放缩的过程
from sklearn import preprocessing

# 标准化 Standardization¶