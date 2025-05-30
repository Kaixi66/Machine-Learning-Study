import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import seaborn as sns

# Boosting方法的基本思想
# 集成学习的泛化误差可分解为：泛化误差 = 偏差**2 + 方差 + 噪声
# 装袋法（Bagging）： 通过并行训练多个独立的弱学习器，利用 “平均化” 效应抵消单个模型的随机波动（方差）。
# 通过串行训练弱学习器，逐步纠正前序模型的错误，使集成模型的预测不断逼近真实标签（降低偏差）。
# 由于专注于偏差降低，Boosting算法们在模型效果方面的突出表现制霸整个弱分类器集成的领域

# 如果说Bagging不同算法之间的核心区别在于靠以不同方式实现“独立性”（随机性），
# 那Boosting的不同算法之间的核心区别就在于上一个弱评估器的评估结果具体如何影响下一个弱评估器的建立过程

# 每个Boosting算法会其以独特的规则自定义集成输出的具体形式
# 集成算法的输出结果往往是关于弱评估器的某种结果的加权平均，
# 其中权重的求解是boosting领域中非常关键的步骤。


# Boosting算法的基本元素与基本流程
# 任意boosting算法的三大基本元素以及boosting算法自适应建模的基本流程：
# 1.损失函数：用以衡量模型预测结果与真实结果的差异
# 2.弱评估器：（一般为）决策树，不同的boosting算法使用不同的建树过程
# 3.综合集成结果：即集成算法具体如何输出集成结果

# AdaBoost (Adaptive Boosting，自适应提升法）
# 主要贡献在于实现了两个变化：
# 1. 首次实现根据之前弱评估器的结果自适应地影响后续建模过程
# 2. 在Boosting算法中，首次实现考虑全部弱评估器结果的输出方式

# 构筑过程:
# 在全样本上建立一棵决策树，根据该决策树预测的结果和损失函数值，
# 增加被预测错误的样本在数据集中的样本权重，并让加权后的数据集被用于训练下一棵决策树。
# 有意地加重“难以被分类正确的样本”的权重，同时降低“容易被分类正确的样本”的权重，
# 而将后续要建立的弱评估器的注意力引导到难以被分类正确的样本上。

# 上一棵决策树的的结果通过影响样本权重、即影响数据分布来影响下一棵决策树的建立，整个过程是自适应的。
# 当全部弱评估器都被建立后，集成算法的输出H(x)等于所有弱评估器输出值的加权平均，加权所用的权重也是在建树过程中被自适应地计算出来的。

# AdaBoost既可以实现分类也可以实现回归
# 参数：
# base_estimator：弱评估器
# 分类问题中只能选择算法，回归问题中只能设置损失函数
# algorithm（分类器专属）：用于指定分类ADB中使用的具体实现方法
# loss（回归器专属）： 用于指定回归ADB中使用的损失函数

from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.datasets import load_digits
#用于分类的数据
data_c = load_digits()
X_c = data_c.data
y_c = data_c.target
# 返回标签数组 y_c 中的所有唯一值
print(np.unique(y_c))  #手写数字数据集，10分类

#用于回归的数据
data_r = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col=0)
X_g = data_r.iloc[:, :-1]
y_g = data_r.iloc[:, -1]
print(X_g.head())

# base_estimator是规定AdaBoost中使用弱评估器的参数。
# ADB分类器的默认弱评估器是最大深度为1的“树桩”，ADB回归器的默认评估器是最大深度为3的“树苗”

# 建立ADB回归器和分类器
clf = ABC(n_estimators=3).fit(X_c, y_c)
reg = ABR(n_estimators=3).fit(X_g, y_g)

print(clf.base_estimator_)
# 当AdaBoost完成分类任务时，弱评估器是分类树，当AdaBoost完成回归任务时，弱评估器是回归树

# 自建评估器
base_estimator = DTC(max_depth=10, max_features=30)
clf = ABC(base_estimator = base_estimator, n_estimators=3).fit(X_c, y_c)
print(clf.base_estimator_)
# 了保证集成算法中的树不一致，AdaBoost会默认消除我们填写在弱评估器中的random_state

# 参数learning_rate
# 集成算法的输出往往都是多个弱评估器的输出结果的加权平均结果。
# 但并不是在所有树建好之后才统一加权求解的，
# 而是在算法逐渐建树的过程当中就随着迭代不断计算出来的
# 可以在权重前面增加学习率，表示为第t棵树加入整体集成算法时的学习率
# 学习率参数控制Boosting集成过程中的增长速度，是相当关键的参数。
#当学习率很大时，集成算法增长得更快，我们所需的n_estimators更少。反之亦然
# 因此boosting算法往往会需要在n_estimators与learning_rate当中做出权衡

# 参数algorithm是针对分类器设置的参数，其中备选项有"SAMME"与"SAMME.R"两个字符串。
# 实现AdaBoost分类的手段：AdaBoost-SAMME与AdaBoost-SAMME.R
# 两者在数学流程上的区别并不大，只不过SAMME是基于算法输出的具体分类结果
# 而SAMME.R则是在SAMME基础上改进过后、基于弱分配器输出的概率值进行计算
# SAMME.R往往能够得到更好的结果，因此sklearn中的默认值是SAMME.R
# sklearn中默认可以输入的base_estimators也需要是能够输出预测概率的弱评估器。
# 实际在预测时，AdaBoost输出的也针对于某一类别的概率。

# 在分类器中，我们虽然被允许选择算法，却不被允许选择算法所使用的损失函数
# 使用了相同的损失函数：二分类指数损失（Exponential Loss Function）与多分类指数损失（Multi-class Exponential loss function）。

# 二分类指数损失
# 根据指数损失的特殊性质，二分类状况下的类别取值只能为-1或1，因此y的取值只能为-1或1。
# L(H(x), y) = e**(-y)(H*(x))
# y 为真实分类
# if H(x) > 0.5
# H*(x) = 1
# if H(x) < 0.5
# H*(x) = -1
# 当算法预测正确时, yH*(x) 符号为正，函数损失很小，反之函数损失大

# 多分类指数损失
# K为总类别数
# L（H（x）, y) = exp( -1/K (y*) * (H*(x)) )
# 简单来说，对于boosting当前的树，会计算一个样本所有可能类别的概率f*（x）
# 最终每一个类别的概率H*（x）等于前面每一个类别概率H*（x）加上当前的f*（x）
# 在softmax中，最大概率变为1，其余为0
# y* 和 H*（x）都是根据多分类具体情况，以及集成算法实际输出H（x）转化出的向量，上标代表当前类别
# 在二分类中，只计算出一类的概率即可判断预测标签。在多分类算法中，算法必须针对所有可能的取值类型都输出概率

# 多分类预测：
clf =  DTC(max_depth=2).fit(X_c, y_c)

# 每一行对应一个样本，每一列则对应该样本的预测标签为某一类别的概率
# 每一个样本的10个概率中，最大概率所对应的类别就是预测类别
print(pd.DataFrame(clf.predict_proba(X_c)).iloc[:5, :])

# 参数Loss：
# 在AdaBoost回归当中，我们能够使用的算法是唯一的，即AdaBoost.R2
# 在R2算法下，我们却可以选择三种损失函数，分别是"linear"（线性）,"square"（平方）,"exponential"（指数）。
# D = sup|H(xi) - yi|, i = 1, 2, ..., N
# 取出1~N号样本中真实值与预测值差距最大的那一组差异来作为D的值。

# R2 Linear loss:
# Li = |H(xi) - yi| / D
# Square Loss:
# Li = |H(xi) - yi|**2 / D**2
# exponent Loss:
# Li = 1 - exp(-|H(xi) - yi| / D)

# 其实线性损失就是我们常说的MAE的变体，平方损失就是MSE的变体，而指数损失也与分类中的指数损失高度相似。
# MAE就是把平方换成绝对值的MSE
# 这些损失函数特殊的地方在于分母D。由于D是所有样本中真实值与预测值差异最大的那一组差异，
# 因此任意样本的在上述线性与平方损失定义下，取值范围都只有[0,1]

# boosting算法的基本规则：
# 依据上一个弱评估器的结果，计算损失函数L，
# 并使用L自适应地影响下一个弱评估器的构建。
# 集成模型输出的结果，受到整体所有弱评估器的影响。