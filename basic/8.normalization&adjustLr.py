# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

a = np.arange(12).reshape(6, 2)
# print(a)


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

# print(maxmin_norm(a))

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
# print(sigmoid(a))



# 2.学习率调度
# 所谓学习率调度，也并不是一个寻找最佳学习率的方法，
# 而是一种伴随迭代进行、不断调整学习率的策略。

#其中一种最为通用的学习率调度方法是学习率衰减法，
#指的是在迭代开始时设置较大学习率，而伴随着迭代进行不断减小学习率。

# 实现了一种指数衰减策略，使学习率每轮降低 5%
lr_lambda = lambda epoch: 0.95 ** epoch
lr_l = []
for i in range(10):
    lr_l.append(lr_lambda(i))
print(np.array(lr_l).reshape(-1, 1))

# 梯度下降
features, labels = arrayGenReg(delta=0.1)
# 深拷贝features用于归一化
features_norm = np.copy(features)

# 归一化处理
features_norm[:, :-1] = z_score(features_norm[:, :-1])
np.random.seed(24) 
n = features.shape[1]
w = np.random.randn(n, 1)
w_norm = np.copy(w)

# 记录迭代过程损失函数取值变化
Loss_l = []
Loss_norm_l = []

# 迭代次数/遍历数据集次数
epoch = 100

for i in range(epoch):
    w = w_cal(features, w, labels, lr_gd, lr = 0.2, itera_times = 1)
    Loss_l.append(MSELoss(features, w, labels))
    w_norm = w_cal(features_norm, w_norm, labels, lr_gd, lr = 0.2, itera_times = 1)
    Loss_norm_l.append(MSELoss(features_norm, w_norm, labels))
    
plt.plot(list(range(epoch)), np.array(Loss_l).flatten(), label='Loss_l')
plt.plot(list(range(epoch)), np.array(Loss_norm_l).flatten(), label='Loss_norm_l')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend(loc = 1)
# plt.show()
print(Loss_l[-1])
print(Loss_norm_l[-1])