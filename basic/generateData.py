import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

num_inputs = 2 # 两个特征
num_examples = 1000 # 一千条数据
np.random.seed(24) #如果你在代码中间设置了种子，那么只有从这一行之后生成的随机数才会受到影响。

# 线性方程组系数,都转为array格式
w_true = np.array([2, -1]).reshape(-1, 1) # -1代表自动计算行数
b_true = np.array(1)

#扰动项系数
delta = 0.01

# 创建数据集的特征和标签取值
features = np.random.randn(num_examples, num_inputs) # randn标准正态分布
labels_true = features.dot(w_true) + b_true  # (1000, 2)*(2, 1) = (1000 * 1)
#真实值再加上扰动项
labels = labels_true + np.random.normal(size = labels_true.shape) * delta # normal：正态分布

#这里subplot把图表划分为1行2个的形式，121代表在1行2个的布局中，设置为第一个。122则为第二个
#scatter为散点图，展示x与y的关系
plt.subplot(121)
plt.scatter(features[:, 0], labels) # 第一个特征和标签的关系
plt.subplot(122)
plt.scatter(features[:, 1], labels) #第二个特征和标签的关系 也就是x和y的关系，这里的y是存在干扰项的，但是也可以看出线性关系
# plt.show()

# 函数默认创造的是y = 2x-x+1的函数
# 可以在其他py文件中用 from generateData import * 来用这里的函数及所有程序
def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程最高项次数
    :return: 生成的特征数组和标签数组
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

# 设置随机数种子
np.random.seed(24)   
# 扰动项取值为0.01
f, l = arrayGenReg(delta=0.01)
# 绘制图像查看结果
plt.subplot(121)
plt.scatter(f[:, 0], l)             # 第一个特征和标签的关系
plt.subplot(122)
plt.scatter(f[:, 1], l)  
plt.show()