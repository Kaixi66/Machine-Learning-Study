# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *


# 二分类交叉熵损失函数,梯度表达式
def logit_gd(X, w, y):
    """
    逻辑回归梯度计算公式
    """
    m = X.shape[0]
    grad = X.T.dot(sigmoid(X.dot(w)) - y) / m
    return grad


# 手动创建分类数据
num_inputs = 2  #数据集特征
num_examples = 500 #样本数
np.random.seed(24)
#np.random.normal(loc均值, scale标准差, size) 用于生成服从正态分布（高斯分布）的随机数
data0 = np.random.normal(4, 2, size=(num_examples, num_inputs))
data1 = np.random.normal(-2, 2, size=(num_examples, num_inputs))
label0 = np.zeros(500)
label1 = np.ones(500)
features = np.concatenate((data0, data1), axis=0)
labels = np.concatenate((label0, label1), axis=0)
#plt.scatter(features[:, 0], features[:, 1], c = labels) # 根据标签选择颜色
# plt.show()
# 均值实际上也就代表着样本分布的中心
# 方差（及标准差）其实表示数据的离散程度，方差越大、数据距离中心点的离散程度就越大

#果我们缩短中心点之间的距离、增加每一个点簇的方差，则能够进一步加深两个点簇彼此交错的情况
# 反之，如果提高中心点间距、减少每个点簇方差，则会减少各点簇彼此交错的情况，并且能够呈现出一条更加清晰的边界
data0 = np.random.normal(2, 4, size=(num_examples, num_inputs))
data1 = np.random.normal(-2, 4, size=(num_examples, num_inputs))

features = np.concatenate((data0, data1), 0)
labels = np.concatenate((label0, label1), 0)
#plt.scatter(features[:, 0], features[:, 1], c=labels)

# 创建分类数据生成器
def arrayGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """分类数据集创建函数。
    
    :param num_examples: 每个类别的数据数量
    :param num_inputs: 数据集特征数量
    :param num_class：数据集标签类别总数
    :param deg_dispersion：数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
    :param bias：建立模型逻辑回归模型时是否带入截距，为True时将添加一列取值全为1的列
    :return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组。
    """
    
    cluster_l = np.empty([num_examples, 1])                            # 每一类标签数组的形状
    mean_ = deg_dispersion[0]                                        # 每一类特征数组的均值的参考值
    std_ = deg_dispersion[1]                                         # 每一类特征数组的方差
    lf = []                                                          # 用于存储每一类特征的列表容器
    ll = []                                                          # 用于存储每一类标签的列表容器
    k = mean_ * (num_class-1) / 2                                    # 每一类特征均值的惩罚因子
    
    for i in range(num_class):
        data_temp = np.random.normal(i*mean_-k, std_, size=(num_examples, num_inputs))     # 生成每一类特征
        lf.append(data_temp)                                                               # 将每一类特征添加到lf中
        labels_temp = np.full_like(cluster_l, i)                                           # 生成某一类的标签
        ll.append(labels_temp)                                                             # 将每一类标签添加到ll中
        
    features = np.concatenate(lf)
    labels = np.concatenate(ll)
    
    if bias == True:
        features = np.concatenate((features, np.ones(labels.shape)), 1)    # 在特征张量中添加一列全是1的列
    return features, labels


# 逻辑回归流程

#创建数据集
np.random.seed(24)
f, l = arrayGenCla(num_class = 2, deg_dispersion = [6, 2], bias = True)          # 离散程度较小
plt.scatter(f[:, 0], f[:, 1], c = l)
np.random.seed()
Xtrain, Xtest, ytrain, ytest = array_split(f, l)

# 数据归一化
mean_ = Xtrain[:, :-1].mean(axis=0)
std_ = Xtrain[:, :-1].std(axis=0)
Xtrain[:, :-1] = (Xtrain[:, :-1] - mean_) / std_
Xtest[:, :-1] = (Xtest[:, :-1] - mean_) / std_

# 定义参数初始值及核心参数
np.random.seed(24)
n = f.shape[1]
w = np.random.randn(n, 1)

batch_size = 50
num_epoch = 20
lr_init = 0.2

lr_lambda = lambda epoch: 0.95 ** epoch

# 模型训练
for i in range(num_epoch):
    w = sgd_cal(Xtrain, w, ytrain, logit_gd, batch_size=batch_size, epoch=1, lr=lr_init*lr_lambda(i))


# 模型输出的概率结果为：
yhat = sigmoid(Xtrain.dot(w))
#转化为分类结果
print(logit_cla(yhat, thr=0.5)[:10])

# 计算预测准确率
def logit_acc(X, w, y, thr=0.5):
    yhat = sigmoid(X.dot(w))
    y_cal = logit_cla(yhat, thr=thr)
    return (y_cal == y).mean()
print(logit_acc(Xtrain, w, ytrain, thr=0.5)) # 训练集精准度
