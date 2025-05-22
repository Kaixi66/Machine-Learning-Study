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

# 树模型的模型形态和建模目标：挖掘有效分类规则并以树状形式呈现
# 一个决策树就是一系列分类规则的叠加

# 结构
# 如果一条边从A点引向B点，则我们这条边对于A点来说是出边、对于B点来说是入边，A节点是B节点的父节点
# 1.根节点（root node）：没有入边，但有零条或者多条出边的点 
# 2.内部点（internal node）：只有一条入边并且有两条或多条出边的点
# 3.叶节点（leaf node）：只有入边但没有出边的点；


# 树模型并不是一个模型，而是一类模型。
# ID3(Iterative Dichotomiser 3) 、C4.5、C5.0决策树
# C4.5都是最通用的决策树算法

# CART全称为Classification and Regression Trees，即分类与回归决策树，同时也被称为C&RT算法
# 拓展了回归类问题的计算流程（此前C4.5只能解决分类问题）
# skelarn中，决策树模型评估器集成的也是CART树模型
# CART树还能够用一套流程同时处理离散变量和连续变量、能够同时处理分类问题和回归问题

# CHAID树
# 个决策树其实是基于卡方检验（Chi-square）的结果来构建的
# CART都只能构建二叉树，而CHAID可以构建多分枝的树