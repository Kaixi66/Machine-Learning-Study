# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

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

import lightgbm as lgb


iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])

X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1),  data["target"], test_size=0.2)

# 模型实例化过程
# 也可以先导入，再调用：
from lightgbm import LGBMClassifier
gbm = LGBMClassifier()
gbm.fit(X_train, y_train)
print(gbm.predict(X_test))

print(np.argmax(gbm.predict_proba(X_test), 1)) #指定按列方向（axis=1）查找最大值的索引，即对每一行单独处理

# 调用accuracy_score函数快速查看最终模型在训练集和测试集上的准确率：
print(accuracy_score(y_train, gbm.predict(X_train)), accuracy_score(y_test, gbm.predict(X_test)))


# LightGBM sklearn API 超参数和使用方法
# 在LGBM的sklearn API中，总共包含四个模型类（也就是四个评估器），
# 分别是lightgbm.LGBMModel、LGBMClassifier 和 LGBMRegressor 以及LGBMRanker：

# LGBMModel是 LightGBM 的基本模型类，它是一个泛型模型类，可以用于各种类型的问题
# 通常，我们不直接使用 LGBMModel，而是使用针对特定任务的子类使用不同的类
# 排序问题则使用LGBMRanker

# LGBMClassifier超参数概览
# LGBMClassifier的决策树剪枝超参数：
# min_split_gain	相当于min_impurity_decrease，再分裂所需最小增益。默认值为 0，表示无限制
# min_child_weight	子节点的最小权重和。默认值为 1e-3。较大的 min_child_weight 可以防止过拟合
# min_child_samples	相当于min_samples_leaf，单个叶子节点上的最小样本数量。默认值为 20。较大的 min_child_samples 可以防止过拟合

# LGBMClassifier的Boosting过程控制超参数解释
# boosting_type: 使用的梯度提升算法类型,默认为 'gbdt'
# 可选项包括 'gbdt'（梯度提升决策树)
# 'dart'（Dropouts meet Multiple Additive Regression Trees）
# 'goss'（Gradient-based One-Side Sampling）
# 'rf'（Random Forest，随机森林）
# 其中GBDT是最常用、且性能最稳定的 boosting 类型
# dart (Dropouts meet Multiple Additive Regression Trees)则是一种结合了 Dropout 和多重加性回归树的方法
# goss(Gradient-based One-Side Sampling)则是前文介绍的梯度的单边采样算法，可以在保持较高精度的同时加速训练过程，适用于大规模数据集，
# 可以在保持较高精度的同时加速训练过程，有些时候精度不如GBDT
# rf则是采用随机森林来进行“Boosting过程”，或者说此时就不再是Boosting，而是替换成了Bagging过程，

# subsample_for_bin
# 该参数表示对连续变量进行分箱时（直方图优化过程）抽取样本的个数，默认取值为200000
# 如果boosting_type选择的是 "goss"，。则subsample_for_bin参数会失去作用，此时无论subsample_for_bin取值多少都不影响最终结果。
# 如果需要控制goss过程，则需要借助top_rate 和 other_rate 这两个参数，但是这两个参数只存在于LGBM原生API中，在sklearn中并没有，因此在使用 LightGBM 的 sklearn API 时，GOSS 采样方法会自动进行调整。


#  LGBMClassifier的特征和数据处理类超参数
# subsample:模型训练时抽取的样本数量，取值范围为 (0, 1]，表示抽样比例，默认为1.0
# subsample_freq:抽样频率，表示每隔几轮进行一次抽样，默认取值为0，表示不进行随机抽样
# colsample_bytree	在每次迭代（树的构建）时，随机选择特征的比例，取值范围为 (0, 1]，默认为1.0

# 其中subsample_for_bin抽样结果用于直方图构建，而subsample抽样结果则是用于模型训练
# 更加关键的是subsample_freq参数，如果subsample_freq=0，则无论subsample取值为多少，模型训练时都不会进行随机抽样
# 不同于subsample是样本抽样，colsample_bytree是每次迭代（每次构建一颗树时）进行的特征抽
# 同时需要注意的是，LGBM和随机森林不同，随机森林是每棵树的每次分裂时都随机分配特征，
# 而LGBM是每次构建一颗树时随机分配一个特征子集，这颗树在成长过程中每次分裂都是依据这个特征子集进行生长

# LGBMClassifier的其他超参数
# objective	指定目标函数，默认为None，会自动判断是二分类还是多分类问题，这里我们也可以手动设置 'binary'(用于二分类问题)或'multiclass'(用于多分类问题)
# class_weight	样本权重设置参数

# LGBMRegressor损失函数
from lightgbm import LGBMRegressor
# LGBMRegressor的损失函数包含了GBDT和XGB的各类回归类损失函数
# 均方误差（MSE, Mean Squared Error）
# 平均绝对误差（MAE, Mean Absolute Error）：通常用于标签存在异常值情况
# Huber损失（Huber Loss）：适用于目标值存在大量异常值或者噪声时。
# Huber损失在预测误差较小时表现为均方误差，在预测误差较大时表现为平均绝对误差，
# 这使得它对异常值具有更好的鲁棒性。
# 通常情况首选MSE，而当标签存在噪声或者异常点时，MAE会表现出更好的泛化能力
# 而Huber则是二者的综合，适用于标签存在少量异常值的数据集，Huber对异常值较为鲁棒，同时又可以保留较好的精度
# Huber 损失的核心缺点在于超参数 δ 敏感和计算复杂度较高

# LightGBM sklearn API进阶使用方法
# 交叉验证
from sklearn.model_selection import cross_val_score
print(cross_val_score(LGBMClassifier(), X_train, y_train))

# 网格搜索
gbm = LGBMClassifier()
param_grid = {'n_estimators': [90, 100, 110]}
LGBM_search = GridSearchCV(estimator=gbm, param_grid=param_grid, n_jobs=10)
LGBM_search.fit(X_train, y_train)
print(LGBM_search.best_score_)
print(LGBM_search.best_params_)

# 构建Pipeline
# Pipeline也是sklearn特有的一项功能，通过Pipeline的构建，我们可以将数据清洗、特征衍生和机器学习封装成一个评估器，进而进行快速的数据处理和预测
# 这里我们尝试将多项式特征衍生、标准化和LGBM三个评估器封装为一个pipeline
LGBM_pipe = make_pipeline(PolynomialFeatures(),
                          StandardScaler(),
                          LGBMClassifier())

LGBM_pipe.fit(X_train, y_train)
print(LGBM_pipe.score(X_test, y_test))


# Pipeline超参数优化与自动机器学习
# 我们尝试围绕上述Pipeline进行超参数优化
print(LGBM_pipe.get_params())
param_pipe = {'polunomialfeatures__degree': [2, 3, 4],
              'lgbmclassifier__n_estimators': [90, 100, 110]}
LGBM_pipe_search = GridSearchCV(estimator=LGBM_pipe, param_grid=param_pipe, n_jobs=10)
LGBM_pipe_search.fit(X_train, y_train)
print(LGBM_pipe_search.best_params_)
print(LGBM_pipe_search.score(X_test, y_test))
