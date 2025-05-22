# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

# Scikit-Learn相关模块
# 评估器类
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据准备
from sklearn.datasets import load_iris

# CART分类树的建模流程与sklearn评估器参数详解

#首先我们来看在特征都为分类变量时、围绕分类问题构建CART的基本过程。

# 划分规则评估指标构建的核心思路:
# 一般来说树模型挑选分类规则的评估指标并不是看每个类别划分后的准确率，
# 而是父节点划分子节点后子节点数据集标签的纯度。
# 决策树生长的方向也就是令每个划分出来的子集纯度越来越高的方向。

# 单独一个数据集的标签纯度衡量指标:

# 用于衡量数据集标签纯度的数值指标一般有三种，分别是分类误差、信息熵和基尼系数
# 分类误差越小，说明数据集标签纯度越高。

# 衡量数据混乱程度的信息熵也可以用于衡量数据集标签纯度
# 信息熵也是在[0,1]之间取值，并且信息熵越小则说明数据集纯度越高。

# 基尼系数通过计算1减去每个类别占比的平方和来作为纯度衡量指标
# 基尼系数在[0, 0.5]范围内取值，并且基尼系数越小表示数据集标签纯度越高
# 在默认情况下，CART树默认选择Gini系数作为评估指标。