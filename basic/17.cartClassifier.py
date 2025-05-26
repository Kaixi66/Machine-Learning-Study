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

# CART分类树的建模流程与sklearn评估器参数详解

#首先我们来看在特征都为分类变量时、围绕分类问题构建CART的基本过程。

# 划分规则评估指标构建的核心思路:
# 一般来说树模型挑选分类规则的评估指标并不是看每个类别划分后的准确率，
# 而是父节点划分子节点后子节点数据集标签的纯度。
# 决策树生长的方向也就是令每个划分出来的子集纯度越来越高的方向。

# 单独一个数据集的标签纯度衡量指标:

# 用于衡量数据集标签纯度的数值指标一般有三种，分别是分类误差、信息熵和基尼系数

# 分类误差越小，说明数据集标签纯度越高。
# 1-np.max([p, 1-p]

# 衡量数据混乱程度的信息熵也可以用于衡量数据集标签纯度
# 信息熵也是在[0,1]之间取值，并且信息熵越小则说明数据集纯度越高。

# 基尼系数通过计算1减去每个类别占比的平方和来作为纯度衡量指标
# 基尼系数在[0, 0.5]范围内取值，并且基尼系数越小表示数据集标签纯度越高
# 在默认情况下，CART树默认选择Gini系数作为评估指标。
# 1 - np.power([p, 1 - p], 2).sum()

# 多个数据集的平均指标
# 量多个数据集作为一个整体时的标签的纯度

# 如果要计算B1、B2整体的基尼系数，
# 则需要在gini_B1、gini_B2的基础上进行各自数据集样本数量占整体数据集比例的加权求和

# 决策树备选规则创建方法
# 要挑选有效分类规则，首先就必须明确如何创建哪些备选规则
# 我们通过寻找这些特征不同取值之间的中间点作为切点，来构造备选规则
# 对于任何一个特征无论是连续型变量还是分类变量，只要有N个取值，就可以创造N-1个划分条件将原数据集划分成两份。
# 无需特别区分两连续变量和离散变量的区别。

# 一般来说对于多个规则，我们首先会计算父节点的基尼系数（Gini(A)），然后计算划分出的两个子节点整体基尼系数（Gini(B)），
# 然后通过对比哪种划分方式能够让二者差值更大，即能够让子节点的基尼系数下降更快，我们就选用哪个规则


# 决策树生长与迭代运算
# 我们根据上一轮的到的结论（数据集划分情况）作为基础条件，
# 来寻找子数据集的最佳分类规则，然后来进行数据集划分，以此往复。既然是迭代运算
# 每一轮迭代实际上是为了更快速的降低基尼系数，也就是希望这一轮划分出来的子数据集纯度尽可能高
# 可以将每一轮迭代过程中父类的基尼系数看成是损失函数值

# 迭代计算的收敛条件
# 基尼系数的减少少于某个值就暂时不做划分
# 最大迭代次数其实就相当于树模型的最高生长层数
# 备选规则都用完了，此时也会停止迭代。

# CART树剪枝
# 决策树生长的层数越多就表示树模型越复杂，此时模型结构风险就越高、模型越容易过拟合

# 树模型的剪枝分为两种
# 其一在模型生长前就限制模型生长，这种方法也被称为预剪枝或者盆栽法
# 另外一种方法则是先让树模型尽可能的生长，然后再进行剪枝，这种方法也被称为后剪枝或者修建法。
# 目前主流的C4.5和CART树都采用的是后剪枝的方法
# CART树则是通过类似正则化的方法在损失函数（基尼系数计算函数）中加入结构复杂度的惩罚因子，来进行剪枝。

# 树模型的结构复杂度其实完全可以用树的层数、每一层分叉的节点数来表示
# 可以不采用这些树模型原生原理的方式来进行剪枝，
# 而是直接将这些决定树模型的复杂度的因素视作超参数，
# 然后通过网格搜索的方式来直接确定泛化能力最强的树模型结构

# CART分类树的Scikit-Learn快速实现方法与评估器参数详解
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# 准备数据集
X = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [1, 1], [1, 2], [1, 2], [2, 1]])
y = np.array([0, 0, 0, 1, 0, 1, 1, 0])

# 调用决策树评估器并进行训练
clf = DecisionTreeClassifier().fit(X, y)
print(clf.score(X, y))


# 树模型分类过程的树状图
# 首先导入tree模块
from sklearn import tree
# 然后调用plot_tree函数进行绘制
plt.figure(figsize=(6, 2), dpi=150)
tree.plot_tree(clf)
# plt.show()

# 重点参数进行详细讲解
# criterion：不纯度衡量指标
# 大多数情况下选择哪个指标并不会实质影响树模型的结构，
# 但相比信息熵，基尼系数复杂度更低、计算速度更快，一般情况下推荐使用基尼系数。

# ccp_alpha：结构风险权重
# ccp是复杂度剪枝（Cost-Complexity Pruning）的简称
# 带有ccp项的剪枝也被称为最小复杂度剪枝，其原理是在决策树的损失函数上加上一个结构风险项，
# 类似于正则化项在线性方程的损失函数中作用相同。
# 和c一样，alpha越大，模型结构惩罚力度就越大，模型结构越简单

# 控制树结构的参数类
# 其一是限制模型整体结构，
# 主要包括限制树深度的max_depth参数和限制叶节点数量的max_leaf_nodes参数
# 所谓树的最大深度，指的是树的最多生长几层，或者除了根节点外总共有几层

# 第二类就是限制树生长的参数，包括从节点样本数量限制树生长的参数，
# 包括min_samples_split、min_samples_leaf两个参数
# 规定一个节点再划分时所需的最小样本数，和一个节点所需的最小样本数。如果不满足min则取消操作，使得模型更简单

# 也有从损失值降低角度出发限制树生长的参数，
# 包括min_impurity_split和min_impurity_decrease参数

# sklearn中在计算父节点和子节点的基尼系数（或信息熵）的差值时，
# 会在计算结果的前面乘以一个父节点占根节点数据量比例的系数作为最终impurity_decrease的结果：
# 防止节点数量过多带来的过拟合

# 控制迭代随机过程的参数类
# 其一是splitter参数，当该参数取值为random时其实是随机挑选分类规则对当前数据集进行划分
# 其二是max_features，该参数可以任意设置最多带入几个特征进行备选规律挖掘
# 也是一种用精度换效率的方式，如此操作肯定会带来模型结果精度的下降