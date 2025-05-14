# 科学计算模块
import numpy as np
import pandas as pd
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
# 自定义模块
from ML_basic_function import *

# 为了避免数据集上下顺序对数据规律的影响，我们会对数据集进行随机切分
# 70 - 80% 为training data，20 - 30%为testing data

A = np.arange(10).reshape(5, 2)
B = np.arange(0, 10, 2).reshape(-1, 1)
np.random.seed(24)
np.random.shuffle(A) #随机打乱data

np.random.seed(24)
np.random.shuffle(B)

# 从行索引的第二，三元素之间切开
# print(np.vsplit(A, [2, ])) # vsplit按照axis = 0，纵向分割。hsplit相反。

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
    Xtrain, Xtest = np.vsplit(features, [split_indices, ])  # X 为特征矩阵，y为标签
    ytrain, ytest = np.vsplit(labels, [split_indices, ])
    return Xtrain, Xtest, ytrain, ytest

f = np.arange(10).reshape(-1, 1) # 要从行向量变为列向量，要不然vsplit会报错
l = np.arange(1, 11).reshape(-1, 1)
# print(array_split(f, l))


# 完整线性回归模训练流程
features, labels = arrayGenReg(delta=0.01)
xtrain, xtest, ytrain, ytest = array_split(features, labels)
w = np.linalg.lstsq(xtrain, ytrain, rcond=-1)[0] 
print(SSELoss(xtest, w, ytest))