import numpy as np
import pandas as pd
import sklearn
import matplotlib as mlp
import matplotlib.pyplot as plt
import time #计时模块time
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_validate, KFold, GridSearchCV

# 随机森林在巨量数据上的增量学习
# sklearn 并未开放接入GPU进行运算的接口
# 我们有两种方式解决这个问题：
# 1.使用其他可以接入GPU的机器学习算法库实现随机森林，比如xgboost
# 2.继续使用sklearn进行训练，但使用增量学习（incremental learning）。

# 增量学习允许算法不断接入新数据来拓展当前的模型，
# 即允许巨量数据被分成若干个子集，分别输入模型进行训练。

# 通常来说，当一个模型经过一次训练之后，如果再使用新数据对模型进行训练，原始数据训练出的模型会被替代掉。
# sklearn的这一覆盖规则是交叉验证可以进行的基础，正因为每次训练都不会受到上次训练的影响，我们才可以使用模型进行交叉验证，否则就会存在数据泄露的情况。
# 增量学习中，原始数据训练的树不会被替代掉，模型会一致记得之前训练过的数据

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv')
def RMSE(cvresult, key):
    return (abs(cvresult[key])**0.5).mean()
X = data.iloc[:, :-1] # 住宅信息
y = data.iloc[:, -1] # salePrice
X_fc = fetch_california_housing().data
y_fc = fetch_california_housing().target

model = RFR(n_estimators=3, warm_start=True) #支持增量学习
model2 = model.fit(X_fc, y_fc)

# 此时，如果让model2继续在kaggle房价数据集X,y上进行训练：
model2 = model2.fit(X.iloc[:,:8],y)
#即便已经对X和y进行了训练，但是model2中对加利福尼亚房价数据集的记忆还在
print((mean_squared_error(y_fc,model2.predict(X_fc)))**0.5)

# 不过，这里存在一个问题：虽然原来的树没有变化，但增量学习看起来并没有增加新的树
# 事实上，对于随机森林而言，我们需要手动增加新的树：

#调用模型的参数，可以通过这种方式修改模型的参数，而不需要重新实例化模型
model2.n_estimators += 2 #增加2棵树，用于增量学习
model2.fit(X.iloc[:, :8], y)
print(model2.estimators_) #原来的树还是没有变化，新增的树是基于新输入的数据进行训练的