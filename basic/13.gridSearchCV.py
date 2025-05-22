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

# 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 超参数调整的核心目的是为了提升模型的泛化能力
# 借助交叉验证，能够提供更有效、更可靠的模型泛化能力的证明。

# 尽管对于机器学习来说超参数众多，但能够对模型的建模结果产生决定性影响的超参数却不多
# 主要采用“经验结合实际”的方式来决定超参数的取值

# 对于一些如正则化系数、特征衍生阶数等，则需要采用一个流程来对其进行调节。
# 而这个流程，一般来说就是进行搜索与枚举，或者也被称为网格搜索（gridsearch）。
# 多个不同参数的不同取值最终将组成一个参数空间（parameter space），
# 在这个参数空间中选取不同的值带入模型进行训练，最终选取一组最优的值作为模型的最终超参数

# 参数空间：
# 需要通过枚举和搜索来进行数值确定的参数取值范围所构成的空间
# 例如对于逻辑回归模型来说，如果选择penalty参数和C来进行搜索调参，则这两个参数就是参数空间的不同维度，
# 而这两个参数的不同取值就是这个参数空间中的一系列点，
# 例如(penalty='l1', C=1)、(penalty='l1', C=0.9)、(penalty='l2', C=0.8)等等，
# 接下来我们就需要从中挑选组一个最优组合。
# 对于逻辑回归来说，我们需要同时带入能够让模型拟合度增加、同时又能抑制模型过拟合倾向的参数来构造参数空间
# 即需要带入特征衍生的相关参数、以及正则化的相关参数。

# 交叉验证与评估指标
# 我们需要找到一个能够基于目前模型建模结果的、能代表模型泛化能力的评估指标
# ROC-AUC或F1-Score， 这两个指标的敏感度要强于准确率
# 如果需要重点识别模型识别1类的能力，则可考虑F1-Score，其他时候更推荐使用ROC-AUC。

# 交叉验证过程 
# 在训练集中进行验证集划分（几折待定）；
# 带入训练集进行建模、带入验证集进行验证，并输出验证集上的模型评估指标；
# 计算多组验证集上的评估指标的均值，作为该超参数下模型最终表现。

# 网格搜索用于寻找最优超参数，而交叉验证确保找到的参数具有泛化能力
# 在大多数情况下，网格搜索（gridsearch）都是和交叉验证（CV）同时出现的
# 由于交叉验证的存在，此时测试集的作用就变成了验证网格搜索是否有效，而非去验证模型是否有效

# 基于Scikit-Learn的网格搜索调参
from sklearn.model_selection import GridSearchCV
# 主要参数分为三类，分别是核心参数、评估参数和性能参数。

# 核心参数，也就是estimator参数和param_grid参数：
# 1.estimator调参对象，某评估器  
# 2.param_grid	参数空间，可以是字典或者字典构成的列表

# 评估参数，涉及到不同参数训练结果评估过程方式的参数 
# 1.scoring	表示选取哪一项评估指标来对模型结果进行评估，默认评估指标是准确率
# 2.refit	表示选择一个用于评估最佳模型的评估指标，
#   然后在最佳参数的情况下整个训练集上进行对应评估指标的计算
# 3. cv	交叉验证的折数， 默认情况下进行5折交叉验证

# 性能参数
# 用于规定调用的核心数和一个任务按照何种方式进行并行运算。
# 1.n_jobs 设置工作时参与计算的CPU核数
# 2.pre_dispatch 多任务并行时任务划分数量
# 在网格搜索中，由于无需根据此前结果来确定后续计算方法，所以可以并行计算

# GridSearchCV使用方法
# step1.创建评估器
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)
clf = LogisticRegression(max_iter=int(1e6), solver='saga')

# step2.创建参数空间
# 此处我们挑选penalty和C这两个参数来进行参数空间的构造。参数空间首先可以是一个字典：
param_grid_simple = {'penalty':['l1', 'l2'], 'C':[1, 0.5, 0.1, 0.05, 0.01]}

# 当超参数之间存在依赖关系（如 penalty='elasticnet' 时才需要 l1_ratio），可通过以下方式处理
param_grid_ra = [
    {'penalty': ['l1', 'l2'], 'C': [1, 0.5, 0.1, 0.05, 0.01]}, 
    {'penalty': ['elasticnet'], 'C': [1, 0.5, 0.1, 0.05, 0.01], 'l1_ratio': [0.3, 0.6, 0.9]}
]
# 即可表示网格搜索在l1+1、l1+0.5...空间与elasticnet+1+0.3、elasticnet+1+0.6...空间同时进行搜索。

# Step3.实例化网格搜索评估器
# 网格搜索的评估器的使用也是先实例化然后进行对其进行训练。
# 此处先实例化一个简单的网格搜索评估器，需要输入此前设置的评估器和参数空间。
search = GridSearchCV(estimator=clf, param_grid=param_grid_simple)

# step4.训练网格搜索评估器
search.fit(X_train, y_train)
# 所谓的训练网格搜索评估器，本质上是在挑选不同的参数组合进行逻辑回归模型训练
# 训练完成后相关结果都保存在search对象的属性中。

# GridSearchCV评估器结果查看
# 返回的就是带有网格搜索挑选出来的最佳参数（超参数）的评估器
print(search.best_estimator_)

# 查看参数
print(search.best_estimator_.coef_)

# 查看训练误差，测试误差
print(search.best_estimator_.score(X_train, y_train), search.best_estimator_.score(X_test, y_test))
# 等价于search.best_estimator_.score
print(search.score(X_train,y_train), search.score(X_test,y_test))

# 查看参数
print(search.best_estimator_.get_params())
print(search.best_score_)
# 指标是交叉验证时验证集准确率的平均值，而不是所有数据的准确率；
# 该指标是网格搜索在进行参数挑选时的参照依据。

print(search.best_params_)
