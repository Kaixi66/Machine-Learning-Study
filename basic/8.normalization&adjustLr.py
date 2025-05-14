# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

a = np.arange(12).reshape(6, 2)
print(a)


# 数据归一化并不会影响数据分布，即不会影响数据的内在规律，
# 只是对数据的数值进行调整,统一量纲。加快梯度下降的收敛速度
# 都是Feature scaling(特征缩放)

# 0-1标准化
# 把所有数据都压缩在1到0之间
def maxmin_norm(X):
    """
    max—min normalization标准化函数
    """
    maxmin_range = X.max(axis=0) - X.min(axis=0)
    return (X - X.min(axis=0)) / maxmin_range

print(maxmin_norm(a))

# Z-Score标准化
# 每一条数据都减去当前列的均值再除以当前列的标准差
# 和0-1标准化不同，Z-Score标准化并不会将数据放缩在0-1之间，
# 而是均匀地分布在0的两侧。类似这种数据也被称为Zero-Centered Data

def z_score(X):
    """
    Z-Score标准化函数
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

# 非线性标准化
# 还有一类使用非线性函数进行归一化操作的方法
# 利用Sigmoid函数对数据集的每一列进行处理，
# 由于Sigmoid函数特性，处理之后的数据也将被压缩到0-1之间。
print(sigmoid(a))