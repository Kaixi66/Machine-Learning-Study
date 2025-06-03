import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import re, pip
import time
import os

# 极限提升树XGBoost(Extreme Gradient Boosting)
# XGBoost是一个以提升树为核心的算法系统，它覆盖了至少3+建树流程、10+损失函数，可以实现各种类型的梯度提升树
# XGBoost天生被设计成支持巨量数据，因此可以自由接入GPU/分布式/数据库等系统
# XGBoost也遵循Boosting算法的基本流程进行建模

# 不同于内嵌在sklearn框架中的其他算法，xgboost是独立的算法库，因此它有一套不同于sklearn代码的原生代码。
# 许多人也会倾向于使用xgboost自带的sklearn接口来实现算法。

from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv' ,index_col=0)
# 指定 CSV 文件的第 1 列（索引从 0 开始）作为 DataFrame 的行索引
X = data.iloc[:, :-1]
y = data.iloc[:, :-1]


#sklearn普通训练代码三步走：实例化，fit，score
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=1412)

xgb_sk = XGBRegressor(random_state=1412) #实例化模型
xgb_sk.fit(Xtrain,Ytrain)
print(xgb_sk.score(Xtest,Ytest)) #默认指标R2

# sklearn交叉验证三步走：实例化，交叉验证，对结果求平均
xgb_sk = XGBRegressor(random_state=1412) #  #实例化模型
# 定义所需的交叉验证方式
cv = KFold(n_splits=5, shuffle=True, random_state=1412)
result_xgb_sk = cross_validate(xgb_sk, X, y, cv=cv
                               , scoring="neg_root_mean_squared_error"
                               ,return_train_score=True
                               , verbose=True
                               , n_jobs=-1)

def RMSE(result, name):
    return abs(result[name].mean())

print(RMSE(result_xgb_sk, "train_score"))
print(RMSE(result_xgb_sk, "test_score"))
# 默认参数下，xgboost模型极度不稳定，并且过拟合的情况非常严重
# 这说明XGBoost的学习能力的确强劲，现有数据量对xgboost来说可能有点不足。
# 可以尝试微调max_depth

xgb_sk = XGBRegressor(max_depth=5, random_state=1412) # 实例化
result_xgb_sk = cross_validate(xgb_sk, X, y, cv=cv
                               , scoring="neg_mean_squared_error"
                               , return_train_score=True
                               , verbose=True
                               , n_jobs=-1)
print(RMSE(result_xgb_sk, "test_score"))

xgb_sk = XGBRegressor(max_depth=5,random_state=1412).fit(X,y)

# 一棵树都是一个单独的Booster提升树，Booster就相当于sklearn中DecisionTreeRegressor，只不过是使用xgboost独有的建树规则进行计算。
#查看一共建立了多少棵树，相当于是n_estimators的取值
print(xgb_sk.get_num_boosting_rounds())
# 我们可以了解到xgboost在sklearn API中都设置了怎样的参数，作为未来调参的参考。
# 对于xgboost分类器，我们还可以调用predict_proba这样的方法来输出概率值
print(xgb_sk.get_params())


# XGBoost回归原生代码实现
# 原生代码必须使用XGBoost自定义的数据结构DMatrix
# 这一数据结构能够保证xgboost算法运行更快，并且能够自然迁移到GPU上运行，
# 类似于列表、数组、Dataframe等结构都不能用于原生代码，
# 因此使用原生代码的第一步就是要更换数据结构。

# 当设置好数据结构后，我们需要以字典形式设置参数。

# 我们将使用xgboost中自带的方法xgb.train或xgb.cv进行训练，

# 训练完毕后，我们可以使用predict方法对结果进行预测。

# 虽然xgboost原生代码库所使用的数据结构是DMatrix，但在预测试输出的数据结构却是普通的数组，因此可以直接使用sklearn中的评估指标

# xgb.train和xgb.cv参数：
# 第一个参数params是需要使用字典定义的参数列表
# 第二个参数dtrain就是Dmatrix结构的训练数据
# 第三个参数num_boost_round相当于n_estimators
# 总共建立多少棵提升树，也就是提升过程中的迭代次数。

import xgboost as xgb
# 1.转换数据格式
data_xgb = xgb.DMatrix(X, y)
# DMatrix会将特征矩阵与标签打包在同一个对象中，且一次只能转换一组数据。
# 并且，我们无法通过索引或循环查看内部的内容，一旦数据被转换为DMatrix，就难以调用或修改了
# 因此，数据预处理需要在转换为DMatrix之前做好

# 如果我们有划分训练集和测试集，则需要分别将训练集和测试集转换为DMatrix：
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3,random_state=1412)
dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)

# 定义所需要输出的参数
params = {"max_depth":5, "seed":1412}
# 一般来说params中会包含至少7~10个参数，如果我们在大型数据上使用xgboost，则可能会涉及到十几个参数或以上，这些参数大多与sklearn中呈现不同的名称

reg = xgb.train(params, data_xgb, num_boost_round=100)
# XGBoost不需要实例化，xgb.train函数包揽了实例化和训练的功能，一行代码解决所有问题。
# 同时，XGBoost在训练时没有区分回归和分类器，它默认是执行回归算
# 除了建树棵树、提前停止这两个关键元素，其他参数基本都被设置在params当中

# 用于预测
y_pred = reg.predict(data_xgb)

# 我们可以使用sklearn中的评估指标进行评估，对回归类算法，xgboost的默认评估指标是RMSE
from sklearn.metrics import mean_squared_error as MSE
MSE(y,y_pred,squared=False) #RMSE


# 使用交叉验证进行训练
params = {"max_depth":5,"seed":1412}
result = xgb.cv(params,data_xgb,num_boost_round=100
                ,nfold=5 #补充交叉验证中所需的参数，nfold=5表示5折交叉验证
                ,seed=1412 #交叉验证的随机数种子，params中的是管理boosting过程的随机数种子
               )

print(result)
# 每次迭代后xgboost会执行5折交叉验证，并收集交叉验证上的训练集RMSE均值、训练集RMSE的标准差、
# 测试集RMSE的均值、测试集RMSE的标准差，这些数据构成了4列数据

# 展示出了随着迭代次数增多，模型表现变化的趋势，因此输出结果可以被用于绘制图像。
plt.figure()  # 创建图像
plt.plot(result["train-rmse-mean"])  # 绘制训练集的平均 RMSE 曲线
plt.plot(result["test-rmse-mean"])   # 绘制测试集的平均 RMSE 曲线
plt.legend(["train", "test"])        # 添加图例，区分训练和测试曲线
plt.title("xgboost 5fold cv")        # 设置标题，表明这是 XGBoost 的 5 折交叉验证结果
plt.show()



# XGBoost分类代码实现
# XGBoost默认会实现回归算法，因此在执行分类的时候，我们需要主动声明算法的类型。
# xgboost是通过当前算法所使用的损失函数来判断任务类型的，即是通过在params中填写的objective参数来判断任务类型。

# 用于回归的参数：
# reg:squarederror：平方损失
# reg:squaredlogerror：平方对数损失

# 用于分类
# binary:logistic：二分类交叉熵损失，使用该损失时predict接口输出概率
# binary:logitraw: 二分类交叉熵损失，predict输出执行sigmoid变化之前的值
# multi:softmax: 多分类交叉熵损失，使用该损失时predict接口输出具体的类别
# multi:softprob：多分类交叉熵，适用该损失时predict接口输出每个样本每个类别下的概率
# 参数objective的默认值为reg:squarederror

#导入2个最简单的分类数据集：乳腺癌数据集与手写数字数据集
from sklearn.datasets import load_breast_cancer, load_digits
# 二分类数据
X_binary = load_breast_cancer().data
y_binary = load_breast_cancer().target
data_binary = xgb.DMatrix(X_binary, y_binary)

# d多分类数据
X_multi = load_digits().data
y_multi = load_digits().target
data_multi = xgb.DMatrix(X_multi, y_multi)

# 设置params，进行训练
# 二分类损失函数一般需要搭配参数eval_matric，用于设置分类的评估指标
# xgboost中默认的二分类指标是对数损失（也就是交叉熵损失logloss）
params1 = {"seed":1412, "objective":"binary:logistic"
           , "eval_metric":"logloss"} # 二分类交叉熵损失

clf_binary = xgb.train(params1, data_binary, num_boost_round=100)

# 对多分类算法来说，除了设置损失函数和评估指标，还需要设置参数num_class(体的标签类别数量)
# 通常来说，算法应该能够根据标签的情况自主判断实际类别为多少

params2 = {"seed":1412, "objective":"multi:softmax"
           , "eval_metric":"mlogloss"  #多分类交叉熵损失 #"merror"
           , "num_class":10}
clf_multi = xgb.train(params2, data_multi, num_boost_round=100)

# 预测与评估
y_pred_binary = clf_binary.predict(data_binary)
y_pred_multi = clf_multi.predict(data_multi)
print(y_pred_binary[:20]) #二分类直接返回概率，不返回类别，需要自己转换
print(y_pred_multi) # #多分类，选择`multi:softmax`时返回具体类别，也可以选择`multi:softprob`返回概率。

from sklearn.metrics import accuracy_score as ACC #当返回具体类别时，可以使用准确率
from sklearn.metrics import log_loss as logloss #当返回概率时，则必须使用交叉熵损失

print(ACC(y_binary,(y_pred_binary > 0.5).astype(int))) #对二分类计算准确率，则必须先转换为类别
print(ACC(y_multi, y_pred_multi))
print(logloss(y_binary,y_pred_binary)) #只有二分类输出了概率，因此可以查看交叉熵损失

# 交叉验证
# 在使用xgb.cv时，我们却需要将评估指标参数写在xgb.cv当中，否则有时候会报出警告
# 如果不介意警告，可以继续将评估指标写在params里的eval_matric参数下
# 在xgb.cv当中，我们需要将评估指标打包成元组，写在参数metrics内部

params2 = {"seed":1412
           , "objective":"multi:softmax" #无论填写什么损失函数都不影响交叉验证的评估指标
           , "num_class":10}
result = xgb.cv(params2,data_multi,num_boost_round=100
                ,metrics = ("mlogloss") #交叉验证的评估指标由cv中的参数metrics决定
                ,nfold=5 #补充交叉验证中所需的参数，nfold=5表示5折交叉验证
                ,seed=1412 #交叉验证的随机数种子，params中的是管理boosting过程的随机数种子
               )

print(result) #返回多分类交叉熵损失

# 参数metrics支持多个评估指标：
params3 = {"seed":1412
           , "objective":"multi:softmax" #无论填写什么损失函数都不影响交叉验证的评估指标
           , "num_class":10}
result = xgb.cv(params3,data_multi,num_boost_round=100
                ,metrics = ("mlogloss","merror")
                ,nfold=5 #补充交叉验证中所需的参数，nfold=5表示5折交叉验证
                ,seed=1412 #交叉验证的随机数种子，params中的是管理boosting过程的随机数种子
               )
print(result) #可以执行多个指标，让输出结果的列数翻倍


# sklearn API的实现
from xgboost import XGBClassifier
# 由于在sklearn API当中，我们明确了正在执行的任务是分类，
# 因此无需再使用损失函数来帮助我们辨别分类的类型了。
# 然而如果是多分类，建议还是在参数中明确所使用的损失函数：

clf = XGBClassifier(objective="multi:softmax"
                    , eval_metric="mlogloss" #设置评估指标避免警告
                    , num_class = 10
                  #  , use_label_encoder=False
                   ) 

clf = clf.fit(X_multi, y_multi)
print(clf.predict(X_multi)) #输出具体数值 - 具体的预测类别
print(clf.score(X_multi,y_multi)) #虽然设置了评估指标，但score接口还是准确率p



# XGBoost 参数
# 1.迭代过程
# 作为Boosting算法，XGBoost的迭代流程与GBDT高度相似，
# 因此XGBoost自然而然也有设置具体迭代次数的参数num_boost_round、
# 学习率参数eta以及设置初始迭代值的base_score。

# 次将本轮建好的决策树加入之前的建树结果时，可以增加参数n，
# 表示为第k棵树加入整体集成算法时的学习率，对标参数eta。
# 该学习率参数控制Boosting集成过程中H(x)的增长速度，是相当关键的参数。
# 当学习率很大时，H(x)增长得更快，我们所需的num_boost_round更少
# boosting算法往往会需要在num_boost_round与eta中做出权衡
# 在XGBoost当中，num_boost_round的默认值为10，eta的默认值为0.3

# 参数base_score
# H0(x)的值在数学过程及算法具体实现过程中都需要进行单独的确定，而这个值就由base_score确定
# 在xgboost中，我们可以对base_score输出任何数值，但并不支持类似于GBDT当中输入评估器的操作。
# 该参数的默认值为0.5
# 当迭代次数足够多、数据量足够大时，调整算法的意义不大，因此我们基本不会调整这个参数。


# 参数max_delta_step
# 这个参数代表了每次迭代时被允许的最大nf(x)
#当参数max_delta_step被设置为0，则说明不对每次迭代的大小做限制，
# 如果该参数被设置为正数C，则代表 nf(x) < C，否则就让算法执行： Hk(x) = Hk-1(x) + c
# 通常来说这个参数是不需要的，但有时候这个参数会对极度不均衡的数据有效。如果样本极度不均衡，那可以尝试在这个参数中设置1~10左右的数。


# xgboost的目标函数
# 与GBDT一样，xgboost的损失函数理论上可以推广到任意可微函数
# 但与GBDT不同的是，xgboost并不只向着损失函数最小化的方向运行，
# xgboost向着令目标函数最小化的方向运行（损失 + 正则化）

# 结构风险又由两部分组成，一部分是控制树结构的，另一部分则是正则项：
# 叶子权重是XGBoost数学体系中非常关键的一个因子，它实际上就是当前叶子j的预测值
# 正则项有两个：使用平方的L2正则项与使用绝对值的L1正则项

# 所有可以自由设置的系数都与结构风险有关，这三个系数也正对应着xgboost中的三个参数：gamma，alpha与lambda
# 参数gamma：乘在一棵树的叶子总量之前，依照叶子总量对目标函数施加惩罚的系数，
# 默认值为0，可填写任何[0, ∞]之间的数字。当叶子总量固定时，gamma越大，结构风险项越大

# lambda：L2正则项系数，放大可控制过拟合
# alpha：L1正则项系数，放大可控制过拟合
# lambda的默认值为1，alpha的默认值为0，因此xgboost默认使用L2正则化。通常来说，我们不会同时使用两个正则化，但我们也可以尝试这么做。


# 当树的结构相对复杂时，`gamma`会比敏感，否则`gamma`可能非常迟钝。
# 当原始标签数值很大、且叶子数量不多时，`lambda`和`alpha`就会敏感，
# 如果原始标签数值很小，这两个参数就不敏感。


# XGBoost的弱评估器
# 在XGBoost当中，我们使用参数booster来控制我们所使用的具体弱评估器。
# 输入"gbtree"表示使用遵循XGBoost规则的CART树，又被称为“XGBoost独有树”

# 输入"dart"表示使用抛弃提升树，DART是Dropout Multiple Additive Regression Tree的简称。
# 这种建树方式受深度学习中的Dropout技巧启发，在建树过程中会随机抛弃一些树的结果，可以更好地防止过拟合。
# 在数据量巨大、过拟合容易产生时，DART树经常被使用，
# 但由于会随机地抛弃到部分树，可能会伤害模型的学习能力，同时可能会需要更长的迭代时间。

# 输入"gblinear"则表示使用线性模型，当弱评估器类型是"gblinear"而损失函数是MSE时，表示使用xgboost方法来集成线性回归。
# 当弱评估器类型是"gblinear"而损失函数是交叉熵损失时，则代表使用xgboost来集成逻辑回归。

# 每一种弱评估器都有自己的params列表，例如只有树模型才会有学习率等参数，只有DART树才会有抛弃率等参数。
# 评估器必须与params中的参数相匹配，否则一定会报错。
# 由于DART树是从gbtree的基础上衍生而来，因此gbtree的所有参数DART树都可以使用。

# dart树
# 在任意以“迭代”为核心的算法当中，我们都面临同样的问题，即最开始的迭代极大程度地影响整个算法的走向，而后续的迭代只能在前面的基础上小修小补
# DART树就可以削弱这些前端树的影响力，大幅提升抗过拟合的能力。

# 参数
# rate_drop: 每一轮迭代时抛弃树的比例, 填写[0.0,1.0]之间的浮点数，默认值为0。

# one_drop：每一轮迭代时至少有one_drop棵树会被抛弃
# 当参数one_drop的值高于rate_drop中计算的结果时，则按照one_drop中的设置执行Dropout。
# 当one_drop的值低于rate_drop的计算结果时，则按rate_drop的计算结果执行Dropout。

# skip_drop：每一轮迭代时可以不执行dropout的概率
# 如果skip_drop说本次迭代不执行Dropout，则忽略one_drop中的设置。

# sample_type：抛弃时所使用的抽样方法
# 填写字符串"uniform"：表示均匀不放回抽样。
# "weighted"：表示按照每棵树的权重进行有权重的不放回抽样。
# 该不放回是指在一次迭代中不放回。每一次迭代中的抛弃是相互独立的，因此每一次抛弃都是从所有树中进行抛弃。

# normalize_type：增加新树时，赋予新树的权重
# 当随机抛弃已经建好的树时，可能会让模型结果大幅度偏移，
# 因此往往需要给与后续的树更大的权重，让新增的、后续的树在整体算法中变得更加重要
# 填写字符串"tree"，表示新生成的树的权重等于所有被抛弃的树的权重的均值。
# "forest"，表示新生成的树的权重等于所有被抛弃的树的权重之和。 

# 算法默认为"tree"，
# 当我们的dropout比例较大，且我们相信希望给与后续树更大的权重时，会选择"forest"模式。

# XGBoost并不会针对每一棵树计算特定的权重。这个树的权重其实指的是整棵树上所有叶子权重之和。

# 当模型容易过拟合时，我们可以尝试让模型使用DART树来减轻过拟合。
# 不过DART树也会带来相应的问题，最明显的缺点就是：
# 用于微调模型的一些树可能被抛弃，微调可能失效
# 由于存在随机性，模型可能变得不稳定，因此提前停止等功能可能也会变得不稳定
# 由于要随机抛弃一些树的结果，在工程上来说就无法使用每一轮之前计算出的Hk-1
# 而必须重新对选中的树结果进行加权求和，可能导致模型迭代变得略微缓慢


# 弱评估器的分枝
# 当参数booster的值被设置为gbtree时，XGBoost所使用的弱评估器是改进后的的CART树，
# 其分枝过程与普通CART树高度一致：
# 向着叶子质量提升/不纯度下降的方向分枝、并且每一层都是二叉树。

# 在CART树的基础上，XGBoost创新了全新的分枝指标：结构分数（Structure Score）与结构分数增益（Gain of Structure Score）（也被叫做结构分数之差），
# 更大程度地保证了CART树向减小目标函数的方向增长。
# XGBoost不接受其他指标作为分枝指标，因此并不存在criterion参数

# 结构分数增益表现为：
# Gain = ScoreL + ScoreR = ScoreP
# Gain = 左节点的结构分数 + 右节点的结构分数 - 父节点的结构分数
# 我们选择增益Gain最大的点进行分枝。
# 结构分数越大越好
# 与信息熵、基尼系数等可以评价单一节点的指标不同，结构分数只能够评估结构本身的优劣，不能评估节点的
# 结构分数也被称为质量分数（quality score）。


# 弱评估器的剪枝
# 参数min_child_weight：可以被广义理解为任意节点上所允许的样本量（样本权重）。
# 很显然，参数min_child_weight越大，模型越不容易过拟合，同时学习能力也越弱。

# 参数gamma：目标函数中叶子数量前的系数，
# 同时也是允许分枝的最低结构分数增益。当分枝时结构增益不足gamma中设置的值，该节点被剪枝。
# gamma在剪枝中的作用就相当于sklearn中的min_impurity_decrease。
# gamma值越大，算法越不容易过拟合，同时学习能力也越弱。

# lambda和alpha：正则化系数，同时也位于结构分数中间接影响树的生长和分枝。
# 当lambda越大，结构分数会越小，参数gamma的力量会被放大，模型整体的剪枝会变得更加严格，
# 同时，由于lambda还可以通过目标函数将模型学习的重点拉向结构风险，因此lambda具有双重扛过拟合能力。

# 当alpha越大时，结构分数会越大，参数gamma的力量会被缩小，模型整体的剪枝会变得更宽松
# 然而，alpha还可以通过目标函数将模型学习的重点拉向结构风险，因此alpha会通过放大结构分数抵消一部分扛过拟合的能力。
# 整体来看，alpha是比lambda更宽松的剪枝方式


# 控制复杂度（二）：弱评估器的训练数据
# XGBoost也继承了GBDT和随机森林的优良传统：可以通过对样本和特征进行抽样来增加弱评估器多样性、从而控制过拟合。
# 样本的抽样：
# 参数subsample：
# 默认为1，可输入(0,1]之间的任何浮点数。
# XGBoost中的样本抽样是不放回抽样，因此不像GBDT或者随机森林那样存在袋外数据的问题，
# 同时也无法抽样比原始数据更多的样本量。
# 因此，抽样之后样本量只能维持不变或变少，如果样本量较少，建议保持subsample=1。

# sampling_method：对样本进行抽样时所使用的抽样方法，默认均匀抽样。
# 输入"uniform"：表示使用均匀抽样，每个样本被抽到的概率一致
# 参数还包含另一种可能的输入"gradient_based"：表示使用有权重的抽样

# 特征的抽样
# 参数colsample_bytree，colsample_bylevel，colsample_bynode
# 所有形似colsample_by*的参数都是抽样比例，可输入(0,1]之间的任何浮点数，默认值都为1。
# 但对XGBoost来说，特征的抽样可以发生在建树之前（由colsample_bytree控制）、
# 生长出新的一层树之前（由colsample_bylevel控制）、
# 或者每个节点分枝之前（由colsample_bynode控制）

# 全特征集 >= 建树所用的特征子集 >= 建立每一层所用的特征子集 >= 每个节点分枝时所使用的特征子集。
# 以上全部参数都需要被写在parmas中，没有任何需要写在xgb.train或xgb.cv中的参数

# 参数early_stopping_rounds：位于xgb.train方法当中。
# 参数evals：位于xgb.train方法当中，用于规定训练当中所使用的评估指标，一般都与损失函数保持一致，也可选择与损失函数不同的指标。该指标也用于提前停止。
# 参数verbosity：用于打印训练流程和训练结果的参数。



# XGBoost的参数空间与超参数优化
# 几乎总是具有巨大影响力：num_boost_round（整体学习能力），eta（整体学习速率）

# 特别说明：
# max_depth在XGBoost中默认值为6，比GBDT中的调参空间略大，但还是没有太多的空间，因此影响力不足。
# GBDT中影响力巨大的max_features对标XGBoost中的colsample_by*系列参数，原则上来说影响力应该非常大，但由于三个参数共同作用，调参难度较高，在只有1个参数作用时效果略逊于max_features。
# min_child_weight与结构分数的计算略微相关，因此有时候会展现出较大的影响力。故而将这个精剪枝参数设置为4星参数。
# 每种任务可选的损失函数不多, 一般损失函数不在调参范围之内
# XGBoost的初始化分数只能是数字，因此当迭代次数足够多、数据量足够大时，起点的影响会越来越小。因此我们一般不会对base_score进行调参。

# 一般不会同时使用三个colsample_by*参数、更不会同时调试三个colsample_by*参数
# 首先，参数colsample_bylevel较为不稳定，不容易把握，
# 因此当训练资源充足时，会同时调整colsample_bytree和colsample_bynode
# 如果计算资源不足，或者优先考虑节约计算时间，则会先选择其中一个参数
# 在这三个参数中，使用bynode在分枝前随机，
# 比使用bytree建树前随机更能带来多样性、更能对抗过拟合，
# 但同时也可能严重地伤害模型的学习能力

# 确认参数空间：
# 对于有界的参数（比如colsample_bynode，subsamples等），
# 或者有固定选项的参数（比如booster,objective），无需确认参数空间

# 对取值较小的参数（例如学习率eta，一般树模型的min_impurity_decrease等），
# 或者通常会向下调整的参数（比如max_depth），一般是围绕默认值向两边展开构建参数空间。

# 对于取值可大可小，且原则上可取到无穷值的参数（num_boost_round，gamma、lambda、min_child_weight等），
# 一般需要绘制学习曲线进行提前探索，或者也可以设置广而稀的参数空间，来一步步缩小范围。

# 其中lambda范围[1,2]之间对模型有影响
# 而gamma在[1e6,1e7]之间才对模型有影响

# 我们可以先规定lambda的参数空间为np.arange(0,3,0.2)，
# 并规定gamma的参数空间为np.arange(1e6,1e7,1e6)。
# 现在我们对剩下2个参数(num_boost_round, min_child_weight)绘制学习曲线进行轻度探索
import xgboost as xgb
data_xgb = xgb.DMatrix(X, y)

import xgboost as xgb
data_xgb = xgb.DMatrix(X,y)

#定义一个函数，用来检测模型迭代完毕后的过拟合情况
def overfitcheck(result):
    return (result.iloc[-1,2] - result.iloc[-1,0])

# num_boost_round 学习曲线绘制, 固定其他参数，只看num_boost
train = []
test = []
option = np.arange(10,300,10)
overfit = []
for i in option:
    params = {"max_depth":5,"seed":1412,"eta":0.1, "nthread":16
             }
    result = xgb.cv(params,data_xgb,num_boost_round=i
                ,nfold=5 #补充交叉验证中所需的参数，nfold=5表示5折交叉验证
                ,seed=1412 #交叉验证的随机数种子，params中的是管理boosting过程的随机数种子
               )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0]) # 训练集上最后一轮迭代的 RMSE 均值
    test.append(result.iloc[-1,2]) # 测试集上最后一轮迭代的 RMSE 均值
plt.plot(option,test);
plt.plot(option,train);
plt.plot(option,overfit);
plt.show()
# 可以看到，num_boost_round大约增长到50左右就不再对模型有显著影响了，我们可以进一步来查看分数到30000以下之后的情况：
plt.plot(option,test)
plt.ylim(20000,30000);
# 100棵树之后损失几乎没有再下降，因此num_boost_round的范围可以定到range(50,200,10)。


# min_child_weight 规定每个节点的样本数，每个样本默认权重为 1，此时等价于最小样本数
# 最佳方案其实是对每个叶子上的样本量进行估计。
print(X.shape)
# 现在总共有样本1460个，在五折交叉验证中每轮训练集共有1460*0.8 = 1168个样本
# 由于CART树是二叉树，我们规定的最大深度为5，因此最多有2**5=32个叶子节点
# 平均每个叶子结点上的样本量大概为1168/32 = 36.5个
# 粗略估计，如果min_child_weight是一个小于36.5的值，就可能对模型造成巨大影响。
# 当然，不排除有大量样本集中在一片叶子上的情况，因此我们可以设置备选范围稍微放大，例如设置为[0,100]来观察模型的结果。

train = []
test = []
option = np.arange(0,100,1)
overfit = []
for i in option:
    params = {"max_depth":5,"seed":1412,"eta":0.1, "nthread":16
              ,"min_child_weight":i
             }
    result = xgb.cv(params,data_xgb,num_boost_round=50
                ,nfold=5 #补充交叉验证中所需的参数，nfold=5表示5折交叉验证
                ,seed=1412 #交叉验证的随机数种子，params中的是管理boosting过程的随机数种子
               )
    overfit.append(overfitcheck(result))
    train.append(result.iloc[-1,0])
    test.append(result.iloc[-1,2])
plt.plot(option,test);
plt.plot(option,train);
plt.plot(option,overfit);
# 很明显，min_child_weight在0~40的范围之内对测试集上的交叉验证损失有较好的抑制作用，
# 因此我们可以将min_child_weight的调参空间设置为range(0,50,2)来进行调参。


# 如此，全部参数的参数空间就确定了:
# num_boost_round	学习曲线探索，最后定为(50,200,10)
# eta	以0.3为中心向两边延展，最后定为(0.05,2.05,0.05)
# booster	两种选项["gbtree","dart"] gbtree: 每一轮迭代生成一棵新的决策树，拟合前一轮的残差（或梯度）, dart通过随机丢弃树，降低模型对特定树的依赖，增强泛化能力
# colsample_bytree	设置为(0,1]之间的值，但由于还有参数bynode，因此整体不宜定得太小，
# 因此定为(0.3,1,0.1)
# colsample_bynode	设置为(0,1]之间的值，定为(0.1,1,0.1)
# gamma	学习曲线探索，有较大可能需要改变，定为(1e6,1e7,1e6)
# lambda	学习曲线探索，定为(0,3,0.2)
# min_child_weight	学习曲线探索，定为(0,50,2)
# max_depth	以6为中心向两边延展，右侧范围定得更大(2,30,2)
# subsample	设置为(0,1]之间的值，定为(0.1,1,0.1)
# objective	两种回归类模型的评估指标["reg:squarederror", "reg:squaredlogerror"]
# rate_drop	如果选择"dart"树所需要补充的参数，设置为(0,1]之间的值(0.1,1,0.1)

# 一般在初次搜索时，我们会设置范围较大、较为稀疏的参数空间，然后在多次搜索中逐渐缩小范围、降低参数空间的维度。




# 基于TEP对XGBoost进行优化
#日常使用库与算法
import pandas as pd
import numpy as np
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
import xgboost as xgb

#导入优化算法
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

# 创建数据
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
data_xgb = xgb.DMatrix(X, y)
# 定义目标函数
def hyperopt_objective(params):

    
    paramsforxgb = {"eta":params["eta"]
                    ,"booster":params["booster"]
                    ,"colsample_bytree":params["colsample_bytree"]
                    ,"colsample_bynode":params["colsample_bynode"]
                    ,"gamma":params["gamma"]
                    ,"lambda":params["lambda"]
                    ,"min_child_weight":params["min_child_weight"]
                    ,"max_depth":int(params["max_depth"])
                    ,"subsample":params["subsample"]
                    ,"objective":params["objective"]
                    ,"rate_drop":params["rate_drop"]
                    ,"nthread":14
                    ,"verbosity":0
                    ,"seed":1412}
    
    result = xgb.cv(paramsforxgb, data_xgb, seed=1412, metrics=("rmse")
                    ,num_boost_round=int(params["num_boost_round"]) )
    return result.iloc[-1, 2]

# 参数空间
param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",50,200,10)
                     ,"eta": hp.quniform("eta",0.05,2.05,0.05)
                     ,"booster":hp.choice("booster",["gbtree","dart"])
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.3,1,0.1)
                     ,"colsample_bynode":hp.quniform("colsample_bynode",0.1,1,0.1)
                     ,"gamma":hp.quniform("gamma",1e6,1e7,1e6)
                     ,"lambda":hp.quniform("lambda",0,3,0.2)
                     ,"min_child_weight":hp.quniform("min_child_weight",0,50,2)
                     ,"max_depth":hp.choice("max_depth",[*range(2,30,2)])
                     ,"subsample":hp.quniform("subsample",0.1,1,0.1)
                     ,"objective":hp.choice("objective",["reg:squarederror","reg:squaredlogerror"])
                     ,"rate_drop":hp.quniform("rate_drop",0.1,1,0.1)
                    }

# 优化函数
def param_hyperopt(max_evals=100):
    # 保存迭代过程
    trials = Trials()

    # 设置提前停止
    early_stop_fn = no_progress_loss(30)

    # 定义代理模型
    params_best = fmin(hyperopt_objective
                       , space = param_grid_simple
                       , algo = tpe.suggest
                       , max_evals = max_evals
                       , verbose = True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                       )
        #打印最优参数，fmin会自动打印最佳分数
    print("\n","\n","best params: ", params_best,
          "\n")
    return params_best, trials

# 训练贝叶斯优化器
# XGBoost中涉及到前所未有多的随机性，因此模型可能表现得极度不稳定，我们需要多尝试几次贝叶斯优化来观察模型的稳定性。因此在这里我们完成了5次贝叶斯优化
params_best, trials = param_hyperopt(100) #由于参数空间巨大，给与100次迭代的空间

# 首先，objective在所有迭代中都被选为"reg:squarederror"，这也是xgboost的默认值，因此不再对该参数进行搜索。
# 同样的。booster参数在5次运行中有4次被选为"dart"，因此基本可以确认对目前的数据使用DART树是更好的选择。
# 同时在参考结果时我们就可以不太考虑第三次搜索的结果，因为第三次搜索是给予普通gbtree给出的结果。

# 对于其他参数，我们则根据搜索结果修改空间范围、增加空间密度，
# 一般让范围向选中更多的一边倾斜，并且减小步长。
# 例如num_boost_round从来没有选到100以下的值，还有一次触顶，两次接近上限，
# 因此可以将原本的范围(50,200,10)修改为(100,300,10)。

param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",100,300,10)
                     ,"eta": hp.quniform("eta",0.05,2.05,0.05)
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.5,1,0.05)
                     ,"colsample_bynode":hp.quniform("colsample_bynode",0.3,1,0.05)
                     ,"gamma":hp.quniform("gamma",5e6,1.5e7,5e5)
                     ,"lambda":hp.quniform("lambda",0,2,0.1)
                     ,"min_child_weight":hp.quniform("min_child_weight",0,10,0.5)
                     ,"max_depth":hp.choice("max_depth",[*range(2,15,1)])
                     ,"subsample":hp.quniform("subsample",0.5,1,0.05)
                     ,"rate_drop":hp.quniform("rate_drop",0.1,1,0.05)
                    }

def hyperopt_objective(params):
    paramsforxgb = {"eta":params["eta"]
                    ,"colsample_bytree":params["colsample_bytree"]
                    ,"colsample_bynode":params["colsample_bynode"]
                    ,"gamma":params["gamma"]
                    ,"lambda":params["lambda"]
                    ,"min_child_weight":params["min_child_weight"]
                    ,"max_depth":int(params["max_depth"])
                    ,"subsample":params["subsample"]
                    ,"rate_drop":params["rate_drop"]
                    ,"booster":"dart"
                    ,"nthred":14
                    ,"verbosity":0
                    ,"seed":1412}
    result = xgb.cv(params,data_xgb, seed=1412, metrics=("rmse")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


# 验证参数
def hyperopt_validation(params):
    paramsforxgb = {"eta":params["eta"]
                    ,"booster":"dart"
                    ,"colsample_bytree":params["colsample_bytree"]
                    ,"colsample_bynode":params["colsample_bynode"]
                    ,"gamma":params["gamma"]
                    ,"lambda":params["lambda"]
                    ,"min_child_weight":params["min_child_weight"]
                    ,"max_depth":int(params["max_depth"])
                    ,"subsample":params["subsample"]
                    ,"rate_drop":params["rate_drop"]
                    ,"nthred":14
                    ,"verbosity":0
                    ,"seed":1412}
    result = xgb.cv(params,data_xgb, seed=1412, metrics=("rmse")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]

bestparams = {'colsample_bynode': 0.45
               , 'colsample_bytree': 1.0
               , 'eta': 0.05
               , 'gamma': 13000000.0
               , 'lambda': 0.5
               , 'max_depth': 6
               , 'min_child_weight': 0.5
               , 'num_boost_round': 150.0
               , 'rate_drop': 0.65
               , 'subsample': 0.8500000000000001} 
start = time.time()
hyperopt_validation(bestparams)