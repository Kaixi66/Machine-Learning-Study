# 科学计算模块
import numpy as np
import pandas as pd

# 回归数据创建函数
def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程最高项次数
    :return: 生成的特征张和标签张量
    """
    
    if bias == True:
        num_inputs = len(w)-1                                                           # 数据集特征个数
        features_true = np.random.randn(num_examples, num_inputs)                       # 原始特征
        w_true = np.array(w[:-1]).reshape(-1, 1)                                        # 自变量系数
        b_true = np.array(w[-1])                                                        # 截距
        labels_true = np.power(features_true, deg).dot(w_true) + b_true                 # 严格满足人造规律的标签
        features = np.concatenate((features_true, np.ones_like(labels_true)), axis=1)    # 加上全为1的一列之后的特征
    else: 
        num_inputs = len(w)
        features = np.random.randn(num_examples, num_inputs) 
        w_true = np.array(w).reshape(-1, 1)         
        labels_true = np.power(features, deg).dot(w_true)
    labels = labels_true + np.random.normal(size = labels_true.shape) * delta
    return features, labels

# SSE计算函数
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

# 数据集随机切分函数
def array_split(features, labels, rate=0.7, random_state=24):
    """
    训练集和测试集切分函数
    
    :param features: 输入的特征张量
    :param labels：输入的标签张量
    :param rate：训练集占所有数据的比例
    :random_state：随机数种子值
    :return Xtrain, Xtest, ytrain, ytest：返回特征张量的训练集、测试集，以及标签张量的训练集、测试集 
    """
    
    np.random.seed(random_state)                           
    np.random.shuffle(features)                             # 对特征进行切分
    np.random.seed(random_state)
    np.random.shuffle(labels)                               # 按照相同方式对标签进行切分
    num_input = len(labels)                                 # 总数据量
    split_indices = int(num_input * rate)                   # 数据集划分的标记指标
    Xtrain, Xtest = np.vsplit(features, [split_indices, ])  
    ytrain, ytest = np.vsplit(labels, [split_indices, ])
    return Xtrain, Xtest, ytrain, ytest