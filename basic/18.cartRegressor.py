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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据准备
from sklearn.datasets import load_iris

# 尽管回归树单独来看是解决回归类问题的模型, 无论是解决回归类问题还是分类问题，CART回归树都是唯一的基础分类器
# 回归树中寻找切分点的方式和分类树的方式相同，都是逐特征寻找不同取值的中间点作为切分点。
# 根据此前介绍，有几个切分点就有几种数据集划分方式、即有同等数量的备选划分规则、或有同等数量的树的生长方式

# 挑选规则
# 分类树中我们是采用基尼系数或者信息熵来衡量划分后数据集标签不纯度下降情况来挑选最佳划分方式，
# 而在回归树中，则是根据划分之后子数据集MSE下降情况来进行最佳划分方式的挑选

# 子数据集整体的MSE计算方法也和CART分类树类似，都是先计算每个子集单独的MSE，
# 然后再通过加权求和的方法来进行计算两个子集整体的MSE

# CART回归树的sklearn快速实现
from sklearn.tree import DecisionTreeRegressor
data = np.array([[1, 1], [2, 3], [3, 3], [4, 6], [5, 6]])
clf = DecisionTreeRegressor().fit(data[:, 0].reshape(-1, 1), data[:, 1])
plt.figure(figsize=(6, 2), dpi=150)
tree.plot_tree(clf)
plt.show()

# criterion是备选划分规则的选取指标，对于CART分类树来说默认基尼系数、可选信息熵，
# 而对于CART回归树来说默认mse