import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import seaborn as sns
import re, pip, conda

# 这类方法会训练多个弱评估器（base estimators）
# 并将它们输出的结果以某种方式结合起来解决一个问题。

# 集成学习可以被分为三个主要研究领域
# 1.模型融合
# 这个领域主要关注强评估器，试图设计出强大的规则来融合强分类器的结果、以获取更好的融合结果
# 投票法Voting、堆叠法Stacking、混合法Blending等

# 2.弱分类器集成
# 这个领域试图设计强大的集成算法、来将多个弱学习器提升成为强学习器。
# 如装袋法bagging，提升法boosting

# 3.混合专家模型（mixture of experts）
# 我们将一个复杂的任务拆解成几个相对简单且更小的子任务，
# 然后针对不同的子任务训练个体学习器（专家），然后再结合这些个体学习器的结果得出最终的输出。


# 我们将对弱分类器集成与模型融合两部分进行详细的说明
# Bagging又称为“装袋法”
# 并行建立多个弱评估器（通常是决策树，也可以是其他非线性算法），并综合多个弱评估器的结果进行输出
# 回归任务时，集成算法的输出结果是弱评估器输出的结果的平均值
# 分类任务时，集成算法的输出结果是弱评估器输出的结果少数服从多数。

# sklearn当中，我们可以接触到两个Bagging集成算法，一个是随机森林（RandomForest），另一个是极端随机树（ExtraTrees），
# 他们都是以决策树为弱评估器的有监督算法，可以被用于分类、回归、排序等各种任务。




# 随机森林RandomForest
# 从提供的数据中随机抽样出不同的子集，用于建立多棵不同的决策树，
# 并按照Bagging的规则对单棵决策树的结果进行集成（回归则平均，分类则少数服从多数）。

# 随机森林回归器和分类器的参数高度一致，只需要讲解其中一个类
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import cross_validate, KFold
#这里我们不再使用cross_val_score，转而使用能够输出训练集分数的cross_validate
#决策树本身就是非常容易过拟合的算法，而集成模型的参数量/复杂度很难支持大规模网格搜索
#因此对于随机森林来说，一定要关注算法的过拟合情况

data = pd.read_csv(r"D:\Pythonwork\2021ML\PART 2 Ensembles\datasets\House Price\train_encode.csv",index_col=0)
data.head() # 用于快速查看数据结构和前几条记录, 默认值为5
data.shape
X = data.iloc[:, :-1] # 基于整数位置索引数据的核心方法
y = data.iloc[:, -1]
print(X.columns.tolist()) # 获取 DataFrame 所有列名（列标签）

reg_f = RFR() # 实例化随机森林
reg_t = DTR() # 实例化决策树
cv = KFold(n_splits=5, shuffle=True, random_state=1412) # 实例化交叉验证
result_t = cross_validate(reg_t #要进行交叉验证的评估器
                          ,X,y #数据
                          ,cv=cv #交叉验证模式
                          ,scoring="neg_mean_squared_error" #评估指标
                          ,return_train_score=True #是否返回训练分数
                          ,verbose=True #是否打印进程
                          ,n_jobs=-1 #线程数
                         )                   

# 在集成学习中，我们衡量回归类算法的指标一般是RMSE（根均方误差），也就是MSE开根号后的结果
# 现实数据的标签往往数字巨大、数据量庞杂，MSE作为平方结果会放大现实数据上的误差
# 因此我们会对平房结果开根号，让回归类算法的评估指标在数值上不要过于夸张

# 随机森林天生就是比决策树更不容易过拟合、泛化能力更强的。


# 弱分类器结构
# 在集成算法当中，控制单个弱评估器的结构是一个重要的课题，因为单个弱评估器的复杂度/结果都会影响全局
# 集成算法中的弱评估器也需要被剪枝

# 分枝标准与特征重要性 criterion与feature_importances_

# 与分类树中的信息熵/基尼系数不同
# 回归树中的criterion可以选择"squared_error"（平方误差），"absolute_error"（绝对误差）以及"poisson"（泊松偏差）

# 作为分枝标准，平方误差比绝对误差更敏感（类似于信息熵比基尼系数更敏感），
# 并且在计算上平方误差比绝对误差快很多。

# 泊松偏差则是适用于一个特殊场景的：当需要预测的标签全部为正整数时，标签的分布可以被认为是类似于泊松分布的。

# 另外，当我们选择不同的criterion之后，决策树的feature_importances_也会随之变化，
# 因为在sklearn当中，feature_importances_是特征对criterion下降量的总贡献量，
# 因此不同的criterion可能得到不同的特征重要性。

# 调节树结构来控制过拟合
# max_depth 最粗犷的剪枝方式
# max_leaf_nodes与min_sample_split
# min_impurity_decrease 最精细的减枝方式，可以根据不纯度下降的程度减掉相应的叶子。默认值为0


# 弱分类器数量
# n_estimators是森林中树木的数量，即弱评估器的数量，
# 在sklearn中默认100，它是唯一一个对随机森林而言必填的参数

# 当n_estimators越大时：
# 模型的复杂程度上升，泛化能先增强再减弱（或不变）
# 模型的学习能力越来越强，在训练集上的分数可能越来越高，过拟合风险越来越高
# 模型需要的算力和内存越来越多
# 模型训练的时间会越来越长

# 因此在调整n_estimators时，我们总是渴望在模型效果与训练难度之间取得平衡，同时我们还需要使用交叉验证来随时关注模型过拟合的情况。


# 弱分类器训练数据
# 决策树分枝：对每个特征决策树都会找到不纯度下降程度最大的节点进行分枝
# 原则上来说，只要给出数据一致、并且不对决策树进行减枝的话，决策树的结构一定是完全相同的。
# 对集成算法来说，平均多棵相同的决策树的结果并没有意义，因此集成算法中每棵树必然是不同的树
# Bagging算法是依赖于随机抽样数据来实现这一点的。

# 随机森林会从提供的数据中随机抽样出不同的子集，用于建立多棵不同的决策树
# 最终再按照Bagging的规则对众多决策树的结果进行集成

# 样本的随机抽样 bootstrap，oob_score，max_samples
# bootstrap参数的输入为布尔值，默认True，控制是否在每次建立决策树之前对数据进行随机抽样
# 在一个含有m个样本的原始训练集中，我们进行随机采样。
# 每次采样一个样本，并在抽取下一个样本之前将该样本放回原始训练集，
# 也就是说下次采样时这个样本依然可能被采集到，这样采集max_samples次
# （行业惯例）抽样数据集的大小与原始数据集一致

# 由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能被忽略。
# 当抽样次数足够多、且原始数据集足够大时，自助集大约平均会包含全数据的63%
# 没参与建模的数据被称为 袋外数据(out of bag data，简写为oob)
# 实际使用随机森林时，袋外数据常常被我们当做验证集使用，
# 所以我们或许可以不做交叉验证、不分割数据集，而只依赖于袋外数据来测试我们的模型即可

# 当boostrap=True时，我们可以使用参数oob_score和max_samples
# oob_score控制是否使用袋外数据进行验证，输入为布尔值，默认为False，如果希望使用袋外数据进行验证，修改为True即可
# max_samples表示自助集的大小，可以输入整数、浮点数或None，默认为None。
# 输入整数m，则代表每次从全数据集中有放回抽样m个样本
#输入浮点数f，则表示每次从全数据集中有放回抽样f*全数据量个样本
#输入None，则表示每次抽样都抽取与全数据集一致的样本量（X.shape[0]）

# oob_score_来查看我们的在袋外数据上测试的结果，遗憾的是我们无法调整oob_score_输出的评估指标，它默认是R2。

reg = RFR(n_estimators=20
          , bootstrap=True #进行随机抽样
          , oob_score=True #按袋外数据进行验证
          , max_samples=500
         ).fit(X,y)

#重要属性oob_score_
reg.oob_score_ #在袋外数据上的R2为83%



# 特征的随机抽样
# 数据抽样还有另一个维度：对特征的抽样。
# 特征进行抽样的参数max_features
# max_features用于控制每次分裂时考虑的特征数量

# 输入整数，表示每次分枝时随机抽取max_features个特征
# 入浮点数，表示每次分枝时抽取round(max_features * n_features)个特征
# 输入"auto"或者None，表示每次分枝时使用全部特征n_features
# 输入"sqrt"，表示每次分枝时使用sqrt(n_features)
# 输入"log2"，表示每次分枝时使用log2(n_features)
# 如果我们想要树之间的差异更大，我们可以设置模式为log2

# 在总数据量有限的情况下，单棵树使用的数据量越大，每一棵树使用的数据就会越相似，
# 每棵树的结构也就会越相似，bagging的效果难以发挥、模型也很容易变得过拟合。
# 因此，当数据量足够时，我们往往会消减单棵树使用的数据量。

# 随机抽样的模式 random_state
# 它控制决策树当中多个具有随机性的流程
#强制」随机抽取每棵树建立时分枝用的特征，抽取的数量可由参数max_features决定
#「强制」随机排序每棵树分枝时所用的特征
#「可选」随机抽取每棵树建立时训练用的样本，抽取的比例可由参数max_samples决定


# 我们在建树的第一步总是会先设置随机数种子为一个固定值，让算法固定下来
# 当数据样本量足够大的时候（数万），变换随机数种子几乎不会对模型的泛化能力有影响，因此在数据量巨大的情况下，我们可以随意设置任意的数值。
# 当数据量较小的时候，我们可以把随机数种子当做参数进行调整，但前提是必须依赖于交叉验证的结果。


# 总结 弱分类器的训练数据	
# bootstrap：是否对样本进行随机抽样
# oob_score：如果使用随机抽样，是否使用袋外数据作为验证集
# max_samples：如果使用随机抽样，每次随机抽样的样本量
# max_features：随机抽取特征的数目
# random_state：控制一切随机模式


# 集成算法的参数空间和网格优化
# 对随机森林来说，我们可以大致如下排列各个参数对算法的影响：
# 1. 几乎总是具有巨大影响：
# n_estimators（整体学习能力）
# max_depth（粗剪枝）
# max_features（随机性）

# 2大部分时间具有影响力：
# max_samples（随机性）
# class_weight（样本均衡

# 3.可能有大影响力， 大部分时候影响力不明显
# min_samples_split（精剪枝）
# min_impurity_decrease（精剪枝）
# max_leaf_nodes（精剪枝）
# criterion（分枝敏感度）

# 4.当数据量足够大时，几乎无影响
# random_state
# ccp_alpha（结构风险）

# 通常在网格搜索当中，我们会考虑所有有巨大影响力的参数、以及1、2个影响力不明显的参数。

# 树的集成模型的参数空间非常难以确定,此时我们就要引入两个工具来帮助我们：
# 1、学习曲线
# 2、决策树对象Tree的属性

# 学习曲线
# 学习曲线是以参数的不同取值为横坐标，模型的结果为纵坐标的曲线。
# 当模型的参数较少、且参数之间的相互作用较小时，我们可以直接使用学习曲线进行调参
# 许多参数对模型的影响是确定且单调的，例如n_estimators，树越多模型的学习能力越强
# 因此我们可能通过学习曲线找到这些参数对模型影响的极限。我们会围绕这些极限点来构筑我们的参数空间。

# n_estimators的学习曲线：
Option = [1, *range(5, 101, 5)]
#生成保存模型结果的arrays
trainRMSE = np.array([])
testRMSE = np.array([])
trainSTD = np.array([])
testSTD = np.array([])


#在参数取值中进行循环
for n_estimators in Option:
    
    #按照当下的参数，实例化模型
    reg_f = RFR(n_estimators=n_estimators,random_state=1412)
    
    #实例化交叉验证方式，输出交叉验证结果
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    result_f = cross_validate(reg_f,X,y,cv=cv,scoring="neg_mean_squared_error"
                              ,return_train_score=True
                              ,n_jobs=-1)
    
    #根据输出的MSE进行RMSE计算
    train = abs(result_f["train_score"])**0.5
    test = abs(result_f["test_score"])**0.5
    
    #将本次交叉验证中RMSE的均值、标准差添加到arrays中进行保存
    trainRMSE = np.append(trainRMSE,train.mean()) #效果越好
    testRMSE = np.append(testRMSE,test.mean())
    trainSTD = np.append(trainSTD,train.std()) #模型越稳定
    testSTD = np.append(testSTD,test.std())

# 随机森林可以调用树的属性
reg_f = RFR(n_estimators=10, random_state=1412)
reg_f = reg_f.fit(X,y)
print(reg_f.estimators_) # 查看森林中所有的树
print(reg_f.estimators_[0]) # 用索引单独提取一棵树

# 属性.max_depth，查看当前树的实际深度
print(reg_f.estimators_[0].tree_.max_depth)

for t in reg_f.estimators_:
    print(t.tree_.max_depth)

reg_f = RFR(n_estimators=100, random_state=1412)
reg_f = reg_f.fit(X, y)
d = pd.Series([], dtype='int64')
for idx, t in enumerate(reg_f.estimators_):
    d[idx] = t.tree_.max_depth
print(d.mean())
print(d.describe())

# 假设现在你的随机森林过拟合，max_depth的最大深度范围设置在[15,25]之间就会比较有效，
# 如果我们希望激烈地剪枝，则可以设置在[10,15]之间。

# 一棵树上的总叶子量
reg_f.estimators_[0].tree_.node_count
#所有树上的总叶子量
for t in reg_f.estimators_:
    print(t.tree_.node_count)

# 根据经验，当决策树不减枝且在训练集上的预测结果不错时，一棵树上的叶子量常常与样本量相当或比样本量更多，
# 算法结果越糟糕，叶子量越少

# 每个节点上的不纯度下降量，为-2则表示该节点是叶子节点
print(reg_f.estimators_[0].tree_.threshold.tolist()[:20])