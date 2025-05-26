import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import time #计时模块time
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_validate, KFold, GridSearchCV


# 使用网格搜索在随机森林上进行调参
# 总是具有巨大影响的参数：n_estimators, max_depth, max_features

# 现在模型正处于过拟合的状态，需要抗过拟合，且整体数据量不是非常多，随机抽样的比例不宜减小，
# 因此我们挑选以下五个参数进行搜索：n_estimators，max_depth，max_features，min_impurity_decrease，criterion。
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv')
def RMSE(cvresult, key):
    return (abs(cvresult[key])**0.5).mean()
X = data.iloc[:, :-1] # 住宅信息
y = data.iloc[:, -1] # salePrice

# step1. 建立benchmark
reg = RFR(random_state=1412)
cv = KFold(n_splits=5, shuffle=True, random_state=1412)
# 测试模型性能。verbose: 输出信息的详细程度
result_pre_adjusted = cross_validate(reg, X, y, cv=cv, scoring="neg_mean_squared_error", return_train_score=True, verbose=True, n_jobs=-1)
print(RMSE(result_pre_adjusted, "train_score"))
print(RMSE(result_pre_adjusted, 'test_score'))

# Step 2.创建参数空间
param_grid_sample = {"criterion":["squared",  "poisson"]
                     , 'n_estimators':[*range(20, 100, 5)]
                     , 'max_depth':[*range(10, 25, 2)]
                     , 'max_features':['log2', 'sqrt', 16, 32, 64, 'auto']
                     , 'min_impurity_decrease':[*np.arange(0, 5, 10)]
                     }


# step3. 实例化用于搜索的评估器，交叉验证评估器和网格搜索评估器
reg = RFR(random_state=1412, verbose=True, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=1412)# random_state就是random_seed
search = GridSearchCV(estimator=reg
                      ,param_grid=param_grid_sample
                      ,scoring="neg_mean_squared_error"
                      ,verbose=True
                      ,cv = cv
                      ,n_jobs=-1)
start = time.time()
search.fit(X, y)
print(search.best_estimator_) # 找到最佳参数模型
print(abs(search.best_score_)**0.5) #最佳RMSE
# 根据grid search创建最佳模型
ad_reg = RFR(n_estimators=85, max_depth=23, max_features=16, random_state=1412)
cv = KFold(n_splits=5,shuffle=True,random_state=1412)
result_post_adjusted = cross_validate(ad_reg,X,y,cv=cv,scoring="neg_mean_squared_error"
                          ,return_train_score=True
                          ,verbose=True
                          ,n_jobs=-1)

RMSE(result_post_adjusted,"train_score")
RMSE(result_post_adjusted,"test_score")