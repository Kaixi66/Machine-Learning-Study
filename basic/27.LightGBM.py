# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# Scikit-Learn相关模块
# 评估器类
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据准备
from sklearn.datasets import load_iris

# LightGBM基本原理与EFB降维方法
# 是一个基于梯度提升决策树（Gradient Boosted Decision Trees，GBDT）的高效、可扩展的机器学习算法
# LGBM算法提出的核心目的是为了解决GBDT算法框架在处理海量数据时计算效率低下的问题
# LGBM以牺牲极小的计算精度为代价，将GBDT的计算效率提升了近20倍
# 由于LGBM“选择性的牺牲精度”从另一个角度来看其实就是抑制模型过拟合，因此在很多场景下，LGBM的算法效果甚至会好于XGB。

# LGBM充分借鉴了XGB提出的一系列提升精度的优化策略，同时在此基础之上进一步提出了一系列的数据压缩和决策树建模流程的优化策略。
# 其中数据压缩方法能够让实际训练的数据量在大幅压缩的同时仍然保持较为完整的信息，
# 而决策树建模流程方面的优化，则是在XGB提出的直方图优化算法基础上进行了大幅优化
