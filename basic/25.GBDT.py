import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import seaborn as sns
# 梯度提升树（Gradient Boosting Decision Tree，GBDT）
# 它即是当代强力的XGBoost、LGBM等算法的基石，也是工业界应用最多、在实际场景中表现最稳定的机器学习算法之一。
# 它融合了Bagging与Boosting的思想、扬长避短
# GBDT中自然也包含Boosting三要素：
# 损失函数L(x, y)：用以衡量模型预测结果与真实结果的差异
# 弱评估器f(x)：（一般为）决策树，不同的boosting算法使用不同的建树过程
# 综合集成结果H(x)：即集成算法具体如何输出集成结果

# GBDT也遵循boosting算法的基本流程进行建模：
# 依据上一个弱评估器的结果，计算损失函数，并使用自适应地影响下一个弱评估器的构建。
# 集成模型输出的结果，受到整体所有弱评估器的影响。

# GBDT在整体建树过程中做出了以下几个关键的改变：
# 弱评估器:
# 无论GBDT整体在执行回归/分类/排序任务，弱评估器一定是回归器。
# GBDT通过sigmoid或softmax函数输出具体的分类结果，但实际弱评估器一定是回归器。

# 损失函数:
# GBDT算法中可选的损失函数非常多

# 拟合残差:
# GBDT通过修改后续弱评估器的拟合目标来直接影响后续弱评估器的结构。
# 在GBDT当中，我们不修改样本权重，
# 但每次用于建立弱评估器的是样本X以及当下集成输出H(x)与真实标签y的差异（y - H(x)）
# 这个差异在数学上被称之为残差（Residual）
# 通过拟合残差来影响后续弱评估器结构。

# 抽样思想:
# GBDT加入了随机森林中随机抽样的思想，在每次建树之前，
# 允许对样本和特征进行抽样来增大弱评估器之间的独立性（也因此可以有袋外数据集）。
# 由于Boosting算法的输出结果是弱评估器结果的加权求和，
# 因此Boosting原则上也可以获得由“平均”带来的小方差红利

# GradientBoosting实现
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_validate, KFold

data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col = 0)
print(data.head())
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

cv = KFold(n_splits=5, shuffle=True, random_state=1412)
def RMSE(result, name):
    return abs(result[name].mean())

# 梯度提升回归树
gbr = GBR(random_state=1412) # 实例化
result_gbdt = cross_validate(gbr, X, y, cv=cv
                             , scoring = "neg_root_mean_squared_error"
                             , return_train_score = True
                             , verbose = True
                             , n_jobs=-1)
print(RMSE(result_gbdt, "train_score"))

# 未调参状态下GBDT的结果是最好的，其结果甚至与经过TPE精密调参后的随机森林结果相差不多

# 梯度提升树分类
X_clf = data.iloc[:, :-2]
y_clf = data.iloc[:, -2]
print(np.unique(y_clf)) # 6 分类
clf = GBC(random_state=1412)
cv = KFold(n_splits=5, shuffle=True, random_state=1412)
result_clf = cross_validate(clf, X_clf, y_clf, cv = cv
                            , return_train_score=True
                            , verbose=True
                            , n_jobs=-1)

print(result_clf["test_score"].mean())

# 梯度提升树的特别参数
# 建立第一个弱评估器时，H0（x）的值在数学过程及算法具体实现过程中都需要进行单独的确定，这一确定过程由参数init确定。
# 参数init：输入计算初始预测结果的估计器对象。
# 评估器必须要具备fit以及predict_proba功能，即我们可以使用决策树、逻辑回归等可以输出概率的模型。
# 如果输出一个已经训练过、且精细化调参后的模型，将会给GBDT树打下坚实的基础。

# 填写为字符串"zero"，则代表令H0 = 0来开始迭代。
# 不填写，或填写为None对象，sklearn则会自动选择类DummyEstimator中的某种默认方式进行预测作为H0的结果
# DummyEstimator类是sklearn中设置的使用超简单规则进行预测的类，
# 其中最常见的规则是直接从训练集标签中随机抽样出结果作为预测标签，也有选择众数作为预测标签等选项。

# 一般在GBDT类的使用过程中，我们不会主动调节参数init
# 在init中输入训练好的模型会加重GBDT的过拟合.通常来说，我们还是会选择"zero"作为init的输入。

# 使用回归器完成分类任务
# GBDT与AdaBoost及随机森林的关键区别之一，是GBDT中所有的弱评估器都是回归树，
# 因此在实际调用梯度提升树完成分类任务时，需要softmax函数或sigmoid函数对回归树输出的结果进行处理。

# 在多分类当中，我们必须求解出所有标签类别所对应的概率，在所有这些概率当中，最大概率所对应的标签才是多分类的预测标签。
# 一般我们在使用softmax函数时，k分类问题则需要向softmax函数输入k个值
# 在使用softmax函数前，我们需要准备好与类别数量相当的H(x)


# GBDT的8种损失函数
# GBDT的功绩之一是将损失函数从有限的指数损失、MSE等推广到了任意可微函数
# 在sklearn中，控制具体损失函数的参数为loss
# 分类器中的loss
# 字符串型，可输入"deviance", "exponential"，默认值="deviance"
# "deviance"直译为偏差，特指逻辑回归的损失函数——交叉熵损失
# "exponential"则特指AdaBoost中使用的指数损失函数

# y w为真实标签，H（x）为集成算法输出结果。p（x）为基于H（x）和sigmoid/softmax函数计算出来的概率值
# 二分类交叉熵损失：L = -( ylog(p(x)) + (1 - y)log(1 - p(x)) )
# log当中输入的一定是概率值。对于逻辑回归来说，概率就是算法的输出，
# 因此我们可以认为逻辑回归中p = H(x)，
# 但对于GBDT来说，p(x) = sigmoid(H(x))这一点一定要注意。

# 二分类指数函数
# L = exp(-y H(x))
# 指数函数的y表示正负类别（-1，1），交叉熵函数的y表示概率分布（0，1）

# 回归数的loss：
# 字符串型，可输入{"squared_error", "absolute_error", "huber", "quantile"}，默认值="squared_error"
# 'squared_error'是指回归的平方误差
# absolute_error'指的是回归的绝对误差
# 'huber'是以上两者的结合。'quantile'则表示使用分位数回归中的弹球损失pinball_loss。

# 如何选择不同的损失函数
# GBDT必须考虑离群值带来的影响。数据中的离群值会极大程度地影响模型地构建
# 离群值（Outlier），又称异常值，是数据集中与其他观测值差异显著、不符合整体模式的数据点。
# 因为boosting特别关注残差，所以Boosting是天生更容易被离群值影响的模型、也更擅长学习离群值的模型。
# 我们也会遇见很多离群值对我们很关键的业务场景：例如，电商中的金额离群用户可能是VIP用户
# 这种状况下我们反而更关注将离群值预测正确

# 当高度关注离群值、并且希望努力将离群值预测正确时，选择平方误差
# MSE作为预测值和真实值差值的平方，会放大离群值的影响，会让算法更加向学习离群值的方向进化
# 努力排除离群值的影响、更关注非离群值的时候，选择绝对误差
# MAE对一切样本都一视同仁，对所有的差异都只求绝对值

# 试图平衡离群值与非离群值、没有偏好时，选择Huber或者Quantileloss

# 弱评估器结构
# 梯度提升树中的弱评估器复杂度
# 在随机森林中我们讲到，森林中任意控制过拟合的参数基本都处于“关闭状态”，例如max_depth的默认值为None，表示不限深度
# 而在AdaBoost中使用的弱分类器都是最大深度为1的树桩或最大深度为3的小树苗，因此基于AdaBoost改进的其他Boosting算法也有该限制
# 即默认弱评估器的最大深度一般是一个较小的数字。
# 对GBDT来说，无论是分类器还是回归器，默认的弱评估器最大深度都为3，
# 因此GBDT默认就对弱评估器有强力的剪枝机制。

# GBDT等Boosting算法处于过拟合状态时，便只能从数据上下手控制过拟合了（例如，使用参数max_features，在GBDT中其默认值为None），
# 毕竟当max_depth已经非常小时，其他精剪枝的参数如min_impurity_decrease一般发挥不了太大的作用。
# 也因此，通常认为Boosting算法比Bagging算法更不容易过拟合

# 弗里德曼均方误差
# criterion是树分枝时所使用的不纯度衡量指标。在sklearn当中，GBDT中的弱学习器是CART树
# 通常来说，我们求解父节点的不纯度与左右节点不纯度之和之间的差值，这个差值被称为不纯度下降量(impurity decrease)
# 对GBDT来说，不纯度的衡量指标有2个：弗里德曼均方误差friedman_mse与平方误差squared_error。
# 大部分时候，使用弗里德曼均方误差可以让梯度提升树得到很好的结果，因此GBDT的默认参数就是Friedman_mse

# 梯度提升树的提前停止
# 无论使用什么算法，只要我们能够找到损失函数上真正的最小值，那模型就达到“收敛”状态，迭代就应该被停止。
# 我们往往是通过给出一个极限资源来控制算法的停止，比如，我们通过超参数设置允许某个算法迭代的最大次数，或者允许建立的弱评估器的个数。
# 作为众多Boosting算法的根基算法，梯度提升树自带了提前停止的相关超参数。

# 根据以下原则来帮助梯度提升树实现提前停止：
# 1.当GBDT已经达到了足够好的效果（非常接近收敛状态），持续迭代下去不会有助于提升算法表现
# 2.GBDT还没有达到足够好的效果（没有接近收敛），但迭代过程中呈现出越迭代算法表现越糟糕的情况
# 3.虽然GBDT还没有达到足够好的效果，但是训练时间太长/速度太慢，我们需要重新调整训练
# 第三种情况可以通过参数verbose打印结果来观察，如果GBDT的训练时间超过半个小时，建树平均时长超出1分钟，我们就可以打断训练考虑重调参数

# 对于1，2两种情况。我们可以规定一个阈值，
# 例如，当连续n_iter_no_change次迭代中，验证集上损失函数的减小值都低于阈值tol，
# 或者验证集的分数提升值都低于阈值tol的时候，我们就令迭代停止。

# 此时，即便我们规定的n_estimators或者max_iter中的数量还没有被用完，我们也可以认为算法已经非常接近“收敛”而将训练停下。
# 这种机制就是提前停止机制Early Stopping。
# 这种机制中，需要设置阈值tol，
# 用于不断检验损失函数下降量的验证集，
# 以及损失函数连续停止下降的迭代轮数n_iter_no_change

# 在GBDT当中，这个流程刚好由以下三个参数控制：
# `validation_fraction`：从训练集中提取出、用于提前停止的验证数据占比，值域为[0,1]。
# `n_iter_no_change`：当验证集上的损失函数值连续n_iter_no_change次没有下降或下降量不达阈值时，
# 则触发提前停止。平时则设置为None，表示不进行提前停止。
# `tol`：损失函数下降的阈值，默认值为1e-4，也可调整为其他浮点数来观察提前停止的情况。

# 需要注意的是，当提前停止条件被触发后，梯度提升树会停止训练，即停止建树。
# 因此，我们使用属性n_estimators_调出的结果很可能不足我们设置的n_estimators
reg1 = GBR(n_estimators=100
           , validation_fraction=0.1 ,n_iter_no_change=3, tol=0.01
           , random_state=1412).fit(X, y)

reg2 = GBR(n_estimators=100, random_state=1412).fit(X, y)

print(reg1.n_estimators_, reg2.n_estimators_)
# reg1提前停止建树，estimators只有63，不足设置的100

# 什么时候使用提前停止：
# 1.当数据量非常大，肉眼可见训练速度会非常缓慢的时候，开启提前停止以节约运算时间
# 2.n_estimators参数范围极广、可能涉及到需要500~1000棵树时，开启提前停止来寻找可能的更小的n_estimators取值
# 3.当数据量非常小，模型很可能快速陷入过拟合状况时，开启提前停止来防止过拟合


# 梯度提升树的袋外数据
# 梯度提升树结合了Boosting和Bagging中的重要思想。
# 梯度提升树在每次建树之前，也允许模型对于数据和特征进行随机有放回抽样，构建与原始数据集相同数据量的自助集。
# 梯度提升树的原理当中，当每次建树之前进行随机抽样时，这种梯度提升树叫做随机提升树（Stochastic Gradient Boosting）。
# 随机提升树输出的结果往往方差更低，但偏差略高。
# 如果我们发现GBDT的结果高度不稳定，则可以尝试使用随机提升树。

# 在GBDT当中，对数据的随机有放回抽样比例由参数subsample确定，
# 当该参数被设置为1时，则不进行抽样，直接使用全部数据集进行训练。
# 当该参数被设置为(0,1)之间的数字时，则使用随机提升树，在每轮建树之前对样本进行抽样。
# 对特征的有放回抽样比例由参数max_features确定

# 存在有放回随机抽样时，当数据量足够大、抽样次数足够多时，
# 大约会有37%的数据被遗漏在“袋外”（out of bag）没有参与训练。
# 使用这37%的袋外数据作为验证数据，对随机森林的结果进行验证。

# 在GBDT当中，这些袋外分数的变化值被储存在属性`oob_improvement_`中，
# 同时，GBDT还会在每棵树的训练数据上保留袋内分数（in-bag）的变化，且储存在属性`train_score_`
# 也就是说，即便在不做交叉验证的情况下，我们也可以简单地通过属性`oob_improvement`与属性`train_score_`来观察GBDT迭代的结果。

reg = GBR(n_estumators=500, learning_rate=0.1
          , subsample=0.3 # m每次建树只抽取30%数据训练
          , random_state=1412).fit(X, y)

# early stopping
reg - GBR(n_estimators=500, learning_rate=0.1
          , tol=1e-6
          , n_iter_no_change=5
          , validation_fraction=0.3
          , subsample=0.3
          , randomsample=0.3
          , random_state=1412).fit(X, y)

# BDT中的树必须一棵棵建立、且后面建立的树还必须依赖于之前建树的结果，因此GBDT很难在某种程度上实现并行
# sklearn并没有提供n_jobs参数给Boosting算法使用,因此GBDT的计算速度难以得到加速

# GBDT的参数空间与超参数优化
# 几乎总是具有巨大影响力的参数：
# n_estimators, learning_rate, max_features
# 参数init对GBDT的影响很大，如果在参数init中填入具体的算法，过拟合可能会变得更加严重

# 抗过拟合、用来剪枝的参数群（max_depth、min_samples_split等）
# 对样本/特征进行抽样的参数们（subsample，max_features等）

# 随机森林中非常关键的max_depth在GBDT中没有什么地位，
# 取而代之的是Boosting中特有的迭代参数学习率learning_rate。

# 在随机森林中，我们总是在意模型复杂度(max_depth)与模型整体学习能力(n_estimators)的平衡
# 在Boosting算法当中，单一弱评估器对整体算法的贡献由学习率参数learning_rate控制，代替了弱评估器复杂度的地位
# Boosting算法天生就假设单一弱评估器的能力很弱,们无法靠降低max_depth的值来大规模降低模型复杂度

# 如果无法对弱评估器进行剪枝，最好的控制过拟合的方法就是增加随机性/多样性，
# 因此max_features和subsample就成为Boosting算法中控制过拟合的核心武器

# 依赖于随机性、而非弱评估器结构来对抗过拟合的特点，让Boosting算法获得了一个意外的优势

# 在GBDT当中，max_depth的调参方向是放大/加深，以探究模型是否需要更高的单一评估器复杂度。
# 相对的在随机森林当中，max_depth的调参方向是缩小/剪枝，用以缓解过拟合。

# 使用基于TPE贝叶斯优化（HyperOpt）对GBDT进行优化
# 一般在初次搜索时，我们会设置范围较大、较为稀疏的参数空间，
# 然后在多次搜索中逐渐缩小范围、降低参数空间的维度。
# 需要注意的是，init参数中需要输入的评估器对象无法被HyperOpt库识别，
# 因此参数init我们只能手动调参。
# 我们就按照init= rf来设置参数

# 基于TPE对GBDT进行优化
#日常使用库与算法
import pandas as pd
import numpy as np
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import cross_validate, KFold

#导入优化算法
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv' ,index_col=0)
# 指定 CSV 文件的第 1 列（索引从 0 开始）作为 DataFrame 的行索引
X = data.iloc[:, :-1]
y = data.iloc[:, :-1]

# 定义参数init需要的算法
rf = RFR(n_estimators=89, max_depth=22, max_features=14,min_impurity_decrease=0
         ,random_state=1412, verbose=False, n_jobs=-1)

# 定义目标函数
def hyperopt_objective(params):
    reg = GBR(n_estimators = int(params["n_estimators"])
              , learning_rate = params["lr"]
              , criterion = params["lr"]
              , loss = params["loss"]
              , max_depth = int(params["max_depth"])
              , max_features = params["max_features"]
              , subsample = params["subsample"]
              , min_impurity_decrease = params["min_impurity_decrease"]
              , init = rf
              , random_state=1412
              , verbose=False)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    validation_loss = cross_validate(reg, X, y
                                     , scoring="neg_mean_squared_error"
                                     , cv=cv
                                     , verbose=False
                                     , n_jobs=-1
                                     , error_score='raise'
                                    )
    
    return np.mean(abs(validation_loss["test_score"]))

# 参数空间
param_grid_simple = {'n_estimators': hp.quniform("n_estimators",25,200,25)
                  ,"lr": hp.quniform("learning_rate",0.05,2.05,0.05)
                  ,"criterion": hp.choice("criterion",["friedman_mse", "squared_error", "mse", "mae"])
                  ,"loss":hp.choice("loss",["squared_error","absolute_error", "huber", "quantile"])
                  ,"max_depth": hp.quniform("max_depth",2,30,2)
                  ,"subsample": hp.quniform("subsample",0.1,0.8,0.1)
                  ,"max_features": hp.choice("max_features",["log2","sqrt",16,32,64,"auto"])
                  ,"min_impurity_decrease":hp.quniform("min_impurity_decrease",0,5,1)
                 }

# 优化函数
def param_hyperopt(max_evals=100):
    
    #保存迭代过程
    trials = Trials()
    
    #设置提前停止
    early_stop_fn = no_progress_loss(100)
    
    #定义代理模型
    params_best = fmin(hyperopt_objective
                       , space = param_grid_simple
                       , algo = tpe.suggest
                       , max_evals = max_evals
                       , verbose=True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                      )
    
    #打印最优参数，fmin会自动打印最佳分数
    print("\n","\n","best params: ", params_best,
          "\n")
    return params_best, trials

# 验证函数

def hyperopt_validation(params):    
    reg = GBR(n_estimators = int(params["n_estimators"])
              ,learning_rate = params["learning_rate"]
              ,criterion = params["criterion"]
              ,loss = params["loss"]
              ,max_depth = int(params["max_depth"])
              ,max_features = params["max_features"]
              ,subsample = params["subsample"]
              ,min_impurity_decrease = params["min_impurity_decrease"]
              ,init = rf
              ,random_state=1412 #GBR中的random_state只能够控制特征抽样，不能控制样本抽样
              ,verbose=False)
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv
                                     ,verbose=False
                                     ,n_jobs=-1
                                    )
    return np.mean(abs(validation_loss["test_score"]))

# 训练贝叶斯优化器
params_best, trials = param_hyperopt(30) #使用小于0.1%的空间进行训练
print(params_best) #注意hp.choice返回的结果是索引，而不是具体的数字

# 根据贝叶斯优化得到的最好超参数
hyperopt_validation({'criterion': "mse",
                     'learning_rate': 0.2,
                     'loss': "squared_error",
                     'max_depth': 24.0,
                     'max_features': "log2",
                     'min_impurity_decrease': 5.0,
                     'n_estimators': 175.0,
                     'subsample': 0.7})

# 不难发现，我们已经得到了历史最好分数，但GBDT的潜力远不止如此。
# 现在我们可以根据第一次训练出的结果缩小参数空间，继续进行搜索
# 我们则根据搜索结果修改空间范围、增加空间密度
# 一般以被选中的值为中心向两边拓展，并减小步长，同时范围可以向我们认为会被选中的一边倾斜

# 例如最大深度max_depth被选为24，我们则将原本的范围(2,30,2)修改为(10,35,1)。
# 同样subsample被选为0.7，我们则将新范围调整为(0.5,1.0,0.05)，依次类推。

param_grid_simple = {'n_estimators': hp.quniform("n_estimators",150,200,5)
                     ,"lr": hp.quniform("learning_rate",0.05,3,0.05)
                     ,"criterion": hp.choice("criterion",["friedman_mse", "squared_error", "mse","mae"])
                     ,"max_depth": hp.quniform("max_depth",10,35,1)
                     ,"subsample": hp.quniform("subsample",0.5,1,0.05)
                     ,"max_features": hp.quniform("max_features",10,30,1)
                     ,"min_impurity_decrease":hp.quniform("min_impurity_decrease",0,5,0.5)
                    }