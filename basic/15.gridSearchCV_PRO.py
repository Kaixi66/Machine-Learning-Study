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

# 构建全域参数搜索空间
# 调参应该是纳入所有对模型结果有影响的参数进行搜索、并且是全流程中的参数来进行搜索
# 先构建一个包含多项式特征衍生的机器学习流、然后围绕这个机器学习流进行参数搜索
np.random.seed(24)
X = np.random.normal(0, 1, size=(1000, 2))
y = np.array(X[:, 0] + X[:, 1] ** 2 < 1.5, int)
np.random.seed(24)
for i in range(200):
    y[np.random.randint(1000)] = 1
    y[np.random.randint(1000)] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
pipe = make_pipeline(PolynomialFeatures(), 
                     StandardScaler(),
                     LogisticRegression(max_iter=int(1e6)))

# 构造参数空间
param_grid = [
    {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['l1'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__solver': ['saga']}, 
    {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['l2'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']},
    {'polynomialfeatures__degree': np.arange(2, 10).tolist(), 'logisticregression__penalty': ['elasticnet'], 'logisticregression__C': np.arange(0.1, 2, 0.1).tolist(), 'logisticregression__l1_ratio': np.arange(0.1, 1, 0.1).tolist(), 'logisticregression__solver': ['saga']}
]


# 优化评估指标选取：核心就是修改scoring参数取值
# 需要更好的验证模型本身泛化能力，建议使用f1-score或者roc-auc
# 如果数据集的各类别并没有明确的差异，在算力允许的情况下，应当优先考虑roc-auc；
# 而如果希望重点提升模型对类别1（或者某类别）的识别能力，则可以优先考虑f1-score作为模型评估指标。


search = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      n_jobs=5)
search.fit(X_train, y_train)
print(search.best_score_)
print(search.best_params_)
# 需要注意的是，上述best_score_属性查看的结果是在roc-auc评估指标下，
# 默认五折交叉验证时验证集上的roc-auc的平均值，

#但如果我们对训练好的评估器使用.socre方法，查看的仍然是pipe评估器默认的结果评估方式，也就是准确率计算结果
print(search.best_estimator_.score(X_train,y_train))
print(search.best_estimator_.score(X_test,y_test))
# 该模型在未来的使用过程中更有可能能够确保一个稳定的预测输出结果（泛化能力更强）。这也是交叉验证和roc-auc共同作用的结果。