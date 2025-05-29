import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 贝叶斯优化方法是当前超参数优化领域的SOTA手段（State of the Art），可以被认为是当前最为先进的优化框架
# 贝叶斯优化算法本身，与贝叶斯优化用于HPO的过程还有区别。

# 主要执行以下几个步骤：
# 1. 定义需要估计的f(x)以及的x定义域
# 2. 取出有限的n个x上的值，求解出这些x对应的f(x)（求解观测值）
# 3. 根据有限的观测值，对函数进行估计（该假设被称为贝叶斯优化中的先验知识），
# 得出该估计f*上的目标值（最大值或最小值）
# 4. 定义某种规则，以确定下一个需要计算的观测点
# 直到假设分布上的目标值达到我们的标准，或者所有计算资源被用完为止
# 以上流程又被称为序贯模型优化（SMBO）

# HPO中的f(X)不能算是严格意义上的黑盒函数。
# 在第3步中，根据有限的观测值、对函数分布进行估计的工具被称为概率代理模型（Probability Surrogate model），
# 这些概率代理模型自带某些假设，他们可以根据廖廖数个观测点估计出目标函数f*的分布
# (包括f*上每个点的取值以及该点对应的置信度）。
# 概率代理模型往往是一些强大的算法，最常见的比如高斯过程、高斯混合模型等等
# 现在最普及的优化库中基本都默认使用基于高斯混合模型的TPE过程。
# 在第4步中用来确定下一个观测点的规则被称为采集函数（Aquisition Function）
# 采集函数衡量观测点对拟合f*所产生的影响，并选取影响最大的点执行下一步观测，因此我们往往关注采集函数值最大的点。
# 最常见的采集函数主要是概率增量PI（Probability of improvement，比如我们计算的频数）、期望增量（Expectation Improvement）、
# 置信度上界（Upper Confidence Bound）、信息熵（Entropy）
# 我们的目标就是在采集函数指导下，让f*尽量接近f(x)

# 介绍如下三个可以实现贝叶斯优化的库：bayesian-optimization，hyperopt，optuna。

#基本工具
import numpy as np
import pandas as pd
import time
import os #修改环境设置

#算法/损失/评估指标等
import sklearn
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold, cross_validate

#优化器
from bayes_opt import BayesianOptimization

import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

import optuna

data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv' ,index_col=0)
# 指定 CSV 文件的第 1 列（索引从 0 开始）作为 DataFrame 的行索引
X = data.iloc[:, :-1]
y = data.iloc[:, :-1]
print(X.head())

# 基于Bayes_opt实现GP优化
# 为数不多至今依然保留着高斯过程优化的优化库
# 缺乏相应的提效/监控功能，对算力的要求较高，因此它往往不是我们进行优化时的第一首选库。

# 1. 定义目标函数
# f(x): 目标函数。 我们希望能够筛选出令模型泛化能力最大的参数组合
# 计算方式需要被明确。在HPO过程中，我们希望能够筛选出令模型泛化能力最大的参数组合，
# 因此f(x)应该是损失函数的交叉验证值或者某种评估指标的交叉验证值。

# bayes_opt库存在三个影响目标函数定义的规则:
# 1. 目标函数的输入必须是具体的超参数，而不能是整个超参数空间，
# 更不能是数据、算法等超参数以外的元素
# 2. 超参数的输入值只能是浮点数，不支持整数与字符串
# 3. bayes_opt只支持寻找f(x)的最大值，不支持寻找最小值。
# 因此当我们定义的目标函数是某种损失时，目标函数的输出需要取负
# 当我们定义的目标函数是准确率,保持原样

# 创建需要被优化的目标函数：
# obejective function输入是超参数
# 输入一定是浮点数
def bayesopt_objective(n_estimators,max_depth,max_features,min_impurity_decrease):
    
    #定义评估器
    #需要调整的超参数等于目标函数的输入，不需要调整的超参数则直接等于固定值
    #默认参数输入一定是浮点数，因此需要套上int函数处理成整数
    reg = RFR(n_estimators = int(n_estimators)
              ,max_depth = int(max_depth)
              ,max_features = int(max_features)
              ,min_impurity_decrease = min_impurity_decrease
              ,random_state=1412
              ,verbose=False #可自行决定是否开启森林建树的verbose
              ,n_jobs=-1)
    
    #定义损失的输出，5折交叉验证下的结果，输出负根均方误差（-RMSE）
    #注意，交叉验证需要使用数据，但我们不能让数据X,y成为目标函数的输入
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv
                                     ,verbose=False
                                     ,n_jobs=-1
                                     ,error_score='raise'
                                     #如果交叉验证中的算法执行报错，则告诉我们错误的理由
                                    )
    
    #交叉验证输出的评估指标是负根均方误差，因此本来就是负的损失
    #目标函数可直接输出该损失的均值
    return np.mean(validation_loss["test_score"])



# 2.定义参数空间
# 在贝叶斯优化中，超参数组合会被输入我们定义好的目标函数中。
# 我们使用字典方式来定义参数空间，其中参数的名称为键，
# 参数的取值范围为值。且任意参数的取值范围为双向闭区间
param_grid_simple = {'n_estimators': (80,100)
                     , 'max_depth':(10,25)
                     , "max_features": (10,20)
                     , "min_impurity_decrease":(0,1)
                    }
# ayes_opt只支持填写参数空间的上界与下界，不支持填写步长等参数，
# 且bayes_opt会将所有参数都当作连续型超参进行处理，
# 因此bayes_opt会直接取出闭区间中任意浮点数作为备选参数
# 所以需要对整数型超参的取值都套上int函数


# 3 定义优化目标函数的具体流程
# 在大部分优化库当中，随机性是无法控制的，即便允许我们填写随机数种子
# 虽然，优化算法无法被复现，但是优化算法得出的最佳超参数的结果却是可以被复现的。

def param_bayes_opt(init_points,n_iter):
    
    #定义优化器，先实例化优化器
    opt = BayesianOptimization(bayesopt_objective #需要优化的目标函数
                               ,param_grid_simple #备选参数空间
                               ,random_state=1412 #随机数种子，虽然无法控制住
                              )
    
    #使用优化器，记住bayes_opt只支持最大化
    opt.maximize(init_points = init_points #抽取多少个初始观测值
                 , n_iter=n_iter #一共观测/迭代多少次
                )
    
    #优化完成，取出最佳参数与最佳分数
    params_best = opt.max["params"]
    score_best = opt.max["target"]
    
    #打印最佳参数与最佳分数
    print("\n","\n","best params: ", params_best,
          "\n","\n","best cvscore: ", score_best)
    
    #返回最佳参数与最佳分数
    return params_best, score_best


# 定义验证函数
def bayes_opt_validation(params_best):
    
    reg = RFR(n_estimators = int(params_best["n_estimators"]) 
              ,max_depth = int(params_best["max_depth"])
              ,max_features = int(params_best["max_features"])
              ,min_impurity_decrease = params_best["min_impurity_decrease"]
              ,random_state=1412
              ,verbose=False
              ,n_jobs=-1)

    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv
                                     ,verbose=False
                                     ,n_jobs=-1
                                    )
    return np.mean(validation_loss["test_score"])

start = time.time()
params_best, score_best = param_bayes_opt(20,280) #初始看20个观测值，后面迭代280次
print('It takes %s minutes' % ((time.time() - start)/60))
validation_score = bayes_opt_validation(params_best)
print("\n","\n","validation_score: ",validation_score)



# 基于HyperOpt实现TPE优化
# Hyperopt优化器是目前最为通用的贝叶斯优化器之一，Hyperopt中集成了包括随机搜索、模拟退火和TPE（Tree-structured Parzen Estimator Approach）等多种优化算法。
# 相比基于高斯过程的贝叶斯优化，基于高斯混合模型的TPE在大多数情况下以更高效率获得更优结果
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss
# 1 定义目标函数
# Hyperopt也有一些特定的规则会限制我们的定义方式
# 1.目标函数的输入必须是符合hyperopt规定的字典
# 2.Hyperopt只支持寻找最小值，不支持寻找最大值
# 因此当我们定义的目标函数是某种正面的评估指标时（如准确率，auc），我们需要对该评估指标取负。
# 如果我们定义的目标函数是负损失，也需要对负损失取绝对值

def hyperopt_objective(params):
    
    #定义评估器
    #需要搜索的参数需要从输入的字典中索引出来
    #不需要搜索的参数，可以是设置好的某个值
    #在需要整数的参数前调整参数类型
    reg = RFR(n_estimators = int(params["n_estimators"])
              ,max_depth = int(params["max_depth"])
              ,max_features = int(params["max_features"])
              ,min_impurity_decrease = params["min_impurity_decrease"]
              ,random_state=1412
              ,verbose=False
              ,n_jobs=-1)
    
    #交叉验证结果，输出负根均方误差（-RMSE）
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv
                                     ,verbose=False
                                     ,n_jobs=-1
                                     ,error_score='raise'
                                    )
    
    #最终输出结果，由于只能取最小值，所以必须对（-RMSE）求绝对值
    #以求解最小RMSE所对应的参数组合
    return np.mean(abs(validation_loss["test_score"]))

# 2 定义参数空间
# 在hyperopt中，我们使用特殊的字典形式来定义参数空间
# 参数空间定义方法应当都为前闭后开区间
# 键只要与目标函数中索引参数的键一致即可
# 值是hyperopt独有的hp函数：
# hp.quniform("参数名称", 下界, 上界, 步长) - 适用于均匀分布的浮点数
# hp.choice("参数名称",["字符串1","字符串2",...]) - 适用于字符串类型，最优参数由索引表示

# 由于hp.choice最终会返回最优参数的索引，容易与数值型参数的具体值混淆，而hp.randint又只能够支持从0开始进行计数，
# 因此我们常常会使用quniform获得均匀分布的浮点数来替代整数。
# 所以在输入目标函数时，则必须确保参数值前存在int函数
param_grid_simple = {'n_estimators': hp.quniform("n_estimators",80,100,1)
                     , 'max_depth': hp.quniform("max_depth",10,25,1)
                     , "max_features": hp.quniform("max_features",10,20,1)
                     , "min_impurity_decrease":hp.quniform("min_impurity_decrease",0,5,1)
                    }



# 定义优化目标函数的具体流程：
# 有了目标函数和参数空间，接下来我们就可以进行优化了。
# 在Hyperopt中，我们用于优化的基础功能叫做fmin
# 我们可以自定义使用的代理模型（参数algo），一般来说我们有tpe.suggest（指代TPE方法）
# 可以通过partial功能来修改算法涉及到的具体参数，
# 包括模型具体使用了多少个初始观测值（参数n_start_jobs），
# 以及在计算采集函数值时究竟考虑多少个样本（参数n_EI_candidates）。
# 还有记录整个迭代过程的trials
# 提前停止参数early_stop_fn中我们一般输入从hyperopt库导入的方法no_progress_loss()
# 这个方法中可以输入具体的数字n，表示当损失连续n次没有下降时，让算法提前停止。

def param_hyperopt(max_evals=100):
    
    #保存迭代过程
    trials = Trials()
    
    #设置提前停止
    early_stop_fn = no_progress_loss(100)
    
    #定义代理模型
    #algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
    params_best = fmin(hyperopt_objective #目标函数
                       , space = param_grid_simple #参数空间
                       , algo = tpe.suggest #代理模型你要哪个呢？
                       #, algo = algo
                       , max_evals = max_evals #允许的迭代次数
                       , verbose=True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                      )
    
    #打印最优参数，fmin会自动打印最佳分数
    print("\n","\n","best params: ", params_best,
          "\n")
    return params_best, trials

# 定义验证函数（非必要）, 验证优化参数的模型
def hyperopt_validation(params):    
    reg = RFR(n_estimators = int(params["n_estimators"])
              ,max_depth = int(params["max_depth"])
              ,max_features = int(params["max_features"])
              ,min_impurity_decrease = params["min_impurity_decrease"]
              ,random_state=1412
              ,verbose=False
              ,n_jobs=-1
             )
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv
                                     ,verbose=False
                                     ,n_jobs=-1
                                    )
    return np.mean(abs(validation_loss["test_score"]))

params_best, trials = param_hyperopt(30)
params_best, trials = param_hyperopt(100) 
hyperopt_validation(params_best)

#打印所有搜索相关的记录
trials.trials[0]

# 由于具有提前停止功能，因此基于TPE的hyperopt优化可能在我们设置的迭代次数被达到之前就停止，
# TPE方法相比于高斯过程计算会更加迅速
# HyperOpt的缺点也很明显，那就是代码精密度要求较高、灵活性较差，略微的改动就可能让代码疯狂报错难以跑通。
# 同时，HyperOpt所支持的优化算法也不够多



# 基于Optuna实现多种贝叶斯优化
# Optuna是目前为止最为成熟、拓展性最强的超参数优化框架
# Optuna的优势在于，它可以无缝衔接到PyTorch
# 也可以与sklearn的优化库scikit-optimize结合使用
import optuna
# 定义目标函数与参数空间
# 在Optuna中，我们并不需要将参数或参数空间输入目标函数，而是需要直接在目标函数中定义参数空间。
# Optuna优化器会生成一个指代备选参数的变量trial。通过变量trail所携带的方法来构造参数空间


def optuna_objective(trial):
    
    #定义参数空间
    n_estimators = trial.suggest_int("n_estimators",80,100,1) #整数型，(参数名称，下界，上界，步长)
    max_depth = trial.suggest_int("max_depth",10,25,1)
    max_features = trial.suggest_int("max_features",10,20,1)
    #max_features = trial.suggest_categorical("max_features",["log2","sqrt","auto"]) #字符型
    min_impurity_decrease = trial.suggest_int("min_impurity_decrease",0,5,1)
    #min_impurity_decrease = trial.suggest_float("min_impurity_decrease",0,5,log=False) #浮点型
    
    #定义评估器
    #需要优化的参数由上述参数空间决定
    #不需要优化的参数则直接填写具体值
    reg = RFR(n_estimators = n_estimators
              ,max_depth = max_depth
              ,max_features = max_features
              ,min_impurity_decrease = min_impurity_decrease
              ,random_state=1412
              ,verbose=False
              ,n_jobs=-1
             )
    
    #交叉验证过程，输出负均方根误差(-RMSE)
    #optuna同时支持最大化和最小化，因此如果输出-RMSE，则选择最大化
    #如果选择输出RMSE，则选择最小化
    cv = KFold(n_splits=5,shuffle=True,random_state=1412)
    validation_loss = cross_validate(reg,X,y
                                     ,scoring="neg_root_mean_squared_error"
                                     ,cv=cv #交叉验证模式
                                     ,verbose=False #是否打印进程
                                     ,n_jobs=-1 #线程数
                                     ,error_score='raise'
                                    )
    #最终输出RMSE
    return np.mean(abs(validation_loss["test_score"]))



# 2 定义优化目标函数的具体流程
# 在HyperOpt当中我们可以调整参数algo来自定义用于执行贝叶斯优化的具体算法，在Optuna中我们也可以。
# 大部分备选的算法都集中在Optuna的模块sampler中，包括我们熟悉的TPE优化、随机网格搜索以及其他各类更加高级的贝叶斯过程
# 在Optuna库中并没有集成实现高斯过程的方法，但我们可以从scikit-optimize里面导入高斯过程来作为optuna中的algo设置

def optimizer_optuna(n_trials, algo):
    
    #定义使用TPE或者GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials = 10, n_ei_candidates = 24)
    elif algo == "GP":
        from optuna.integration import SkoptSampler
        import skopt
        algo = SkoptSampler(skopt_kwargs={'base_estimator':'GP', #选择高斯过程
                                          'n_initial_points':10, #初始观测点10个
                                          'acq_func':'EI'} #选择的采集函数为EI，期望增量
                           )
    
    #实际优化过程，首先实例化优化器
    study = optuna.create_study(sampler = algo #要使用的具体算法
                                , direction="minimize" #优化的方向，可以填写minimize或maximize
                               )        
    #开始优化，n_trials为允许的最大迭代次数
    #由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
    study.optimize(optuna_objective #目标函数
                   , n_trials=n_trials #最大迭代次数（包括最初的观测值的）
                   , show_progress_bar=True #要不要展示进度条呀？
                  )
    
    #可直接从优化好的对象study中调用优化的结果
    #打印最佳参数与最佳损失值
    print("\n","\n","best params: ", study.best_trial.params,
          "\n","\n","best score: ", study.best_trial.values,
          "\n")
    
    return study.best_trial.params, study.best_trial.values

# 但当参数空间较小时，Optuna库在迭代中容易出现抽样BUG，即Optuna会持续抽到曾经被抽到过的参数组合
# 持续报警告说"算法已在这个参数组合上检验过目标函数了
# 一旦出现这个Bug，那当下的迭代就无用了，因为已经检验过的观测值不会对优化有任何的帮助
# 因此对损失的优化将会停止。如果出现该BUG，则可以增大参数空间的范围或密度。或者使用如下的代码令警告关闭：
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
best_params, best_score = optimizer_optuna(10,"GP") #默认打印迭代过程

# 基于高斯过程的贝叶斯优化是比基于TPE的贝叶斯优化运行更加缓慢的,
# 在TPE模式下，其运行速度与HyperOpt的运行速度高度接近。 