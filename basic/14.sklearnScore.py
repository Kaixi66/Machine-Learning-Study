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

# 多分类F1-Score评估指标
from sklearn.metrics import precision_score, recall_score, f1_score
y_true = np.array([1, 0, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 1, 0, 1])
print(precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))

# 我们令1类标签为0、2类标签为1、3类标签为2，则上述数据集真实标签为：
y_true = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0, 2])
# 最终分类预测结果
y_pred = np.array([0, 1, 0, 2, 2, 1, 2, 2, 0, 2])

# 三个类别的TP和FN
tp1 = 2
tp2 = 2
tp3 = 3

fn1 = 1
fn2 = 1
fn3 = 1

# 有两种计算recall的方法，其一是先计算每个类别的recall，然后求均值：
re1 = 2/3
re2 = 2/3
re3 = 3/4
print(np.mean([re1, re2, re3]))
# 这也就是average参数取值为macro时的计算结果：
print(recall_score(y_true, y_pred, average='macro'))

# 如果上述手动实现过程不求均值，而是根据每个类别的数量进行加权求和，
# 则就是参数average参数取值为weighted时的结果：
print(re1 * 3/10 + re2 * 3/10 + re3 * 4/10)
print(recall_score(y_true, y_pred, average='weighted'))
# 另外一种计算方法，那就是先计算整体的TP和FN，然后根据整体TP和FN计算recall：
# 也就是average参数取值micro时的计算结果
tp = tp1 + tp2 + tp3
fn = fn1 + fn2 + fn3
print(tp / (tp+fn))
print(recall_score(y_true, y_pred, average='micro'))
# 如果是样本不平衡问题（如果是要侧重训练模型判别小类样本的能力的情况下）、
# 则应排除weighted参数，以避免赋予大类样本更高的权重
# 大部分情况三个参数差别不大
# 新版roc-auc不支持micro计算，只支持macro。在之后统一采用macro
# 在roc-auc中，macro指标并不利于非平衡样本的计算

# 多分类ROC-AUC评估指标
# 适用于不平衡数据集或需要综合衡量模型在不同阈值下表现的场景。
from sklearn.metrics import roc_auc_score

# 多分类问题
# 如果是围绕逻辑回归多分类评估器来进行结果评估，则建议roc-auc和逻辑回归评估器的multi_class参数都选择ovr。
# 1类
y_pred_1 = np.array([0.8, 0.2, 0.5, 0.2, 0.3, 0.1, 0.3, 0.3, 0.9, 0.3])
# 2类
y_pred_2 = np.array([0.2, 0.6, 0.3, 0, 0.2, 0.8, 0.2, 0.3, 0, 0.1])
# 3类
y_pred_3 = np.array([0, 0.2, 0.2, 0.8, 0.5, 0.1, 0.5, 0.4, 0.1, 0.6])
#合并
y_pred = np.concatenate([y_pred_1.reshape(-1, 1), y_pred_2.reshape(-1, 1), y_pred_3.reshape(-1, 1)], 1)
#真实值
y_true = np.array([0, 1, 2, 2, 0, 1, 1, 2, 0, 2])

print(roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr'))
print(roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr'))