import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import time #计时模块time
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_validate, KFold, GridSearchCV

#打包成函数供后续使用
#评估指标RMSE
def RMSE(cvresult,key):
    return (abs(cvresult[key])**0.5).mean()

#计算参数空间大小
def count_space(param):
    no_option = 1
    for i in param:
        no_option *= len(param[i])
    print(no_option)
    
#在最优参数上进行重新建模验证结果
def rebuild_on_best_param(ad_reg):
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    result_post_adjusted = cross_validate(ad_reg,X,y,cv=cv,scoring="neg_mean_squared_error"
                                          ,return_train_score=True
                                          ,verbose=True
                                          ,n_jobs=-1)
    print("训练RMSE:{:.3f}".format(RMSE(result_post_adjusted,"train_score")))
    print("测试RMSE:{:.3f}".format(RMSE(result_post_adjusted,"test_score")))


# 当代超参数优化算法主要可以分为：
# 基于网格的各类搜索（Grid）
# 基于贝叶斯优化的各类优化算法（Baysian）
# 基于梯度的各类优化（Gradient-based）
# 基于种群的各类优化（进化算法，遗传算法等）

# 各类网格搜索方法与基于贝叶斯的优化方法是最为盛行的，
# 贝叶斯优化方法甚至可以被称为是当代超参数优化中的SOTA模型

# 网格搜索的理论极限与缺点
# 极端情况下，当参数空间穷尽了所有可能的取值时，
# 网格搜索一定能够找到损失函数的最小值所对应的最优参数组合，
# 但是，参数空间越大，网格搜索所需的算力和时间也会越大
# 我们将介绍三种基于网格进行改进的超参数优化方法


# 随机网格搜索RandomizedSearchCV
# 网格搜索优化方法主要包括两类，其一是调整搜索空间，其二是调整每次训练的数据。
# 调整参数空间的具体方法，是放弃原本的搜索中必须使用的全域超参数空间，
# 改为挑选出部分参数组合，构造超参数子空间，并只在子空间中进行搜索。

# 随机抽取参数子空间并在子空间中进行搜索的方法叫做随机网格搜索
# 随机网格搜索得出的最小损失与枚举网格搜索得出的最小损失很接近。
# 在这一次迭代中随机抽取1组参数进行建模，下一次迭代再随机抽取1组参数进行建模，
# 由于这种随机抽样是不放回的，因此不会出现两次抽中同一组参数的问题。

from sklearn.model_selection import RandomizedSearchCV
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv')
X = data.iloc[:, :-1] # 住宅信息
y = data.iloc[:, -1] # salePrice

#参数空间
param_grid_simple = {"criterion": ["squared_error","poisson"]
                     , 'n_estimators': [*range(20,100,5)]
                     , 'max_depth': [*range(10,25,2)]
                     , "max_features": ["log2","sqrt",16,32,64,"auto"]
                     , "min_impurity_decrease": [*np.arange(0,5,10)]
                    }

reg = RFR(random_state=1412, verbose=True, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=1412)

# 计算全域参数空间大小，这是我们能够抽样的最大值
print(count_space(param_grid_simple))
# 定义随机搜索
search = RandomizedSearchCV(estimator=reg
                            , param_distributions=param_grid_simple
                            , n_iter=800
                            , scoring="neg_mean_squared_error"
                            , verbose = True
                            , cv = cv
                            , random_state=1412
                            , n_jobs=-1)
search.fit(X, y)
print(search.best_estimator_)
print(abs(search.best_score_)**0.5)

# 根据最优参数重建模型
ad_reg = RFR(max_depth=24, max_features=16, min_impurity_decrease=0,
                      n_estimators=85, n_jobs=-1, random_state=1412,
                      verbose=True)
rebuild_on_best_param(ad_reg)
# 抽样出的子空间可以一定程度上反馈出全域空间的分布，
# 且子空间相对越大（含有的参数组合数越多），子空间的分布越接近全域空间的分布
# 不过，由于随机网格搜索计算更快，所以在相同计算资源的前提下，
# 我们可以对随机网格搜索使用更大的全域空间，因此随机搜索可能得到比网格搜索更好的效果

# 除了可以容忍更大的参数空间之外，随机网格搜索还可以接受连续性变量作为参数空间的输入。
# 对于网格搜索来说，只能使用组合好的参数组合点
# 损失函数位于两组参数之间的最低点很不幸的
# 而随机搜索却可以接受“分布”作为输入。

import scipy #使用scipy来帮助我们建立分布
# scipy.stats.uniform(loc=1,scale=100)
# loc：起点位置，scale：分布的范围
# 生成的区间：[loc, loc+scale)
# scipy直接生成一个分布对象，在分布上的随机取值是由随机搜索自己决定的。
# 给出的n_iter越大，任意参数的分布上可能被取到的点就越多

# 连续型搜索更适用于学习率，C，alpha这样的参数（无上限，以浮点数为主），
# 随机森林的参数中最接近这个定义的是min_impurity_decrease，
# 表示决策树在分枝时可以容忍的最小的不纯度下降量。
param_grid_simple = {'n_estimators': [*range(80,100,1)]
                     , 'max_depth': [*range(10,25,1)]
                     , "max_features": [*range(10,20,1)]
                     , "min_impurity_decrease": scipy.stats.uniform(0,50)
                    }
reg = RFR(random_state=1412, verbose=True, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=1412)

# 定义随机搜索
search = RandomizedSearchCV(estimator = reg
                            , param_distributions = param_grid_simple
                            , n_iter = 1536
                            , scoring = "neg_mean_squared_error"
                            , verbose = True
                            , cv = cv
                            , random_state = 412
                            , n_jobs = 12)
rebuild_on_best_param(search.best_estimator_)



# 对半网格搜索HalvingSearchCV
# 面对枚举网格搜索过慢的问题，sklearn中呈现了两种优化方式：其一是调整搜索空间，其二是调整每次训练的数据。
# 调整搜索空间的方法就是随机网格搜索，而调整每次训练数据的方法就是对半网格搜索。

# 当子集的分布越接近全数据集的分布，同一组参数在子集与全数据集上的表现越有可能一致
# 如果一组参数在子集上表现不好，我们也不会信任这组参数在全数据集上的表现。

# 对半网格搜索算法设计了一个精妙的流程，可以很好的权衡子集的大小与计算效率问题
# 1.首先从全数据集中无放回随机抽样出一个很小的子集，并在子集上验证全部参数组合的性能。
# 根据子集上的验证结果，淘汰评分排在后1/2的那一半参数组合
# 2.然后，从全数据集中再无放回抽样出一个比上一个子集大一倍的子集，
# 并验证剩下的那一半参数组合的性能。根据验证结果，淘汰评分排在后1/2的参数组合
# 持续循环
# 在迭代过程中，用于验证参数的数据子集是越来越大的，而需要被验证的参数组合数量是越来越少的：
# 当备选参数组合只剩下一组，或剩余可用的数据不足，循环就会停下 
# 1/n C <= 1或者nS > 总体样本量
# 最终选择出的参数组合一定是在所有子集上都表现优秀的参数组合

# 对半网格搜索的局限性
# 如果最初的子集与全数据集的分布差异巨大的化，在对半搜索开头的前几次迭代中，
# 就可能筛掉许多对全数据集D有效的参数，因此对半网格搜索最初的子集一定不能太小。
# 对半网格搜索在小型数据集上的表现往往不如随机网格搜索与普通网格搜索
# 在大型数据上，对半网格搜索则展现出运算速度和精度上的巨大优势。

# 对半网格的实现：
# 因此在对半网格搜索实现时，我们使用一组拓展的房价数据集，有2w9条样本。
data2 = pd.read_csv("'/Users/kaixi/Downloads/datasets/House Price/big_train.csv'")
X = data2.iloc[:, :-1]
y = data2.iloc[:, -1]

import re
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, HalvingGridSearchCV, cross_validate, RandomizedSearchCV
# 在导入该类的时候需要同时导入用以开启对半网格搜索的辅助功能enable_halving_search_cv
# 参数:
# factor: 每轮迭代中新增的样本量的比例，同时也是每轮迭代后留下的参数组合的比例
# 如果factor=3时，下一轮迭代的样本量会是上一轮的3倍，每次迭代后有1/3的参数组合被留下。
# 该参数通常取3时效果比较好。

# resource: 设置每轮迭代中增加的验证资源的类型，输入为字符串。默认是样本量，输入为"n_samples"

# min_resource: 首次迭代时，用于验证参数组合的样本量r0
# 可以输入正整数，或两种字符串"smallest","exhaust"。
# 输入正整数n，表示首次迭代时使用n个样本。
# 输入"smallest"，则根据规则计算r0
# 输入"exhaust"，则根据迭代最后一轮的最大可用资源倒退r0

param_grid_simple = {"criterion": ["squared_error","poisson"]
                     , 'n_estimators': [*range(20,100,5)]
                     , 'max_depth': [*range(10,25,2)]
                     , "max_features": ["log2","sqrt",16,32,64,"auto"]
                     , "min_impurity_decrease": [*np.arange(0,5,10)]
                    }

# aggressive_elimination
# 输入布尔值，默认False。当数据总样本量较小，不足以支撑循环直到只剩下最后一组备选参数时，可以打开该参数。
# 参数设置为True时，会重复使用首次迭代时的样本量，直到剩下的数据足以支撑样本量的增加直到只剩下最后一组备选参数
# 参数设置为False时，以全部样本被用完作为搜索结束的指标

# 在调参时，如果我们希望参数空间中的备选组合都能够被充分验证，则迭代次数不能太少（例如，只迭代3次），因此factor不能太大。
# 但如果factor太小，又会加大迭代次数，同时拉长整个搜索的运行时间。

# 一般在使用对半网格搜索时，需考虑以下三个点：
# 1、min_resources的值不能太小，且在全部迭代过程结束之前，我们希望使用尽量多的数据
# 2、迭代完毕之后，剩余的验证参数组合不能太多，10以下最佳，如果无法实现，则30以下也可以接受
# 3. 迭代次数不能太多，否则时间可能会太长

#建立回归器、交叉验证
reg = RFR(random_state=1412,verbose=True,n_jobs=-1)
cv = KFold(n_splits=5,shuffle=True,random_state=1412)
#定义对半搜索
search = HalvingGridSearchCV(estimator=reg
                            ,param_grid=param_grid_simple
                            ,factor=1.5
                            ,min_resources=500
                            ,scoring = "neg_mean_squared_error"
                            ,verbose = True
                            ,random_state=1412
                            ,cv = cv
                            ,n_jobs=-1)
search.fit(X,y)
#验证最佳参数组合的效力
rebuild_on_best_param(search.best_estimator_)