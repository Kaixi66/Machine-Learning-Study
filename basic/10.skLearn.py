# 科学计算模块
import numpy as np
import pandas as pd
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics
# 自定义模块
from ML_basic_function import *

import sklearn
#sklearn默认接收的对象类型是数组，
# 即无论是特征矩阵还是标签数组，最好都先转化成array对象类型再进行输入。

# sklearn核心对象类型：评估器（estimator）
# 将评估器就理解成一个个机器学习模型，
# 而sklearn的建模过程最核心的步骤就是围绕着评估器进行模型的训练


np.random.seed(24)
features, labels = arrayGenReg(delta=0.01)

# sklearn中的线性回归评估器LinearRegression实际上是在sklearn包中的linear_model模块下
# 因为是类，实例化
model = sklearn.linear_model.LinearRegression()
# 需要输入上述数据对该模型进行训练。此时我们无需在输入的特征矩阵中加入一列全都是1的列：
X = features[:, :2]
y = labels

# 当fit方法执行完后，即完成了模型训练，此时model就相当于一个参数个数、参数取值确定的线性方程
model.fit(X, y)

# 查看自变量参数
print(model.coef_)
#查看模型截距
print(model.intercept_)

# 对比最小二乘法手动计算
print(np.linalg.lstsq(features, labels, rcond=-1)[0])

# 可以使用model中的predict方法进行预测
print("\n")
print(model.predict(X)[:10])
print("\n")
print(y[:10])

# 在metrics模块下导入MSE计算函数
test = sklearn.metrics.mean_squared_error(model.predict(X), y)
print(test)


# estimator中Parameters部分就是当前模型超参数的相关说明
model1 = sklearn.linear_model.LinearRegression(fit_intercept=False) 
#不带截距的线性方程组
print('\n', model1.get_params())

# 超参数可以更改
model1.set_params(fit_intercept=True)

model.set_params(fit_intercept=False)
# 还没给x和y时，模型参数还不会进行修改
print(model.coef_, model.intercept_) #还带有截距
model.fit(X, y)
print("\n", model.coef_, model.intercept_)


# sklearn创建数据集
# 鸢尾花数据
from sklearn.datasets import load_iris
iris_data = load_iris()
print(iris_data)
print("\n", type(iris_data)) # bunch类型，类似字典类型的对象
#data为特征矩阵，target为标签
print(iris_data.data[:10])
print("\n", iris_data.target[:10])

# 返回各列名称
print("\n", iris_data.feature_names)

#返回标签各类名称
print("\n", iris_data.target_names[:10])

# 只返回特征矩阵和标签数组这两个对象
X, y = load_iris(return_X_y=True)
print(X[:10])
print(y[:10])

# 创建dataframe对象
iris_dataFrame = load_iris(as_frame=True)
print(iris_dataFrame.frame)

# sklearn中数据集切分方法
from sklearn.model_selection import train_test_split

# sklearn中的数据标准化与归一化
# 从功能上划分，sklearn中的归一化其实是分为标准化（Standardization）
# 和归一化（Normalization）两类
# 其中，此前所介绍的Z-Score标准化和0-1标准化，都属于Standardization的范畴
# Normalization则特指针对单个样本（一行数据）利用其范数进行放缩的过程
from sklearn import preprocessing

# 标准化 Standardization 
# 包括Z-Score标准化，也包括0-1标准化
# 可以通过实用函数来进行标准化处理，同时也可以利用评估器来执行标准化过程

# preprocessing模块下的scale函数进行快速的Z-Score标准化处理。
X = np.arange(9).reshape(3, 3)
print(X)
print(preprocessing.scale(X))


# 通过评估器的方法进行数据标准化，其实是一种更加通用的选择。
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = np.arange(15).reshape(5, 3)
X_train, X_test = train_test_split(X)
print(X_train, "\n", X_test)

# 此处标准化的评估器的训练结果对输入数据的相关统计量进行了汇总计算
# 计算了输入数据的均值、标准差等统计量，后续将用这些统计量对各数据进行标准化计算
scaler.fit(X_train)

# 查看数据各列的标准差，均值，方差, 总共有效的训练条数（列数）
# 但尚未对任何数据进行修改
print(scaler.scale_, scaler.mean_, scaler.var_, scaler.n_samples_seen_)

# 利用训练的均值和方差对训练集进行标准化处理
print(scaler.transform(X_train))

# 利用训练的均值和方差对测试集进行标准化处理
# 保证训练和测试数据的标准化过程一致，避免数据泄漏（Data Leakage）和模型性能虚高
print('\n', scaler.transform(X_test))

# 使用fit_transform对输入数据进行直接fit + transform：
scaler = StandardScaler()
print('\n', scaler.fit_transform(X_train))

print(scaler.transform(X_test))

# 0-1标准化的函数实现方法
print("\n\n")
print(X)
print(preprocessing.minmax_scale(X))

# 0-1标准化的评估器实现方法
from sklearn.preprocessing import MinMaxScaler
print("\n\n")
scaler = MinMaxScaler()
print(scaler.fit_transform(X))
print(scaler.data_max_, scaler.data_min_) # 和前面一样以列的视角



# 归一化 Normalization
# 和标准化不同，sklearn中的归一化特指将单个样本（一行数据）放缩为单位范数
# （1范数或者2范数为单位范数）的过程
# 用来衡量样本之间相似性

# 1-范数为各分量的绝对值之和
# 2-范数则为各分量的平方和再开根

# 而sklearn中的Normalization过程，实际上就是将每一行数据视作一个向量，
# 然后用每一行数据去除以该行数据的1-范数或者2-范数。
# 具体除以哪个范数，以preprocessing.normalize函数中输入的norm参数为准。

# 1-范数单位化
print("\n\n",preprocessing.normalize(X, norm='l1'))

# 2-范数单位化
print("\n\n", preprocessing.normalize(X, norm='l2'))

#调用评估器来实现
from sklearn.preprocessing import Normalizer
normlize = Normalizer()
print("\n\n", normlize.fit_transform(X))
normlize = Normalizer(norm='l1')
print(normlize.fit_transform(X))

# 正则化，往往指的是通过在损失函数上加入参数的1-范数或者2-范数的过程，
# 该过程能够有效避免模型过拟合。


# 进行逻辑回归模型训练
from sklearn.linear_model import LogisticRegression
#数据准备
X, y = load_iris(return_X_y=True)

# 实例化模型，使用默认参数
# 此处设置两个参数，一个是最大迭代次数，另一个则是MvM策略
clf_test = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf_test.fit(X, y)

#查看线性方程系数
print(clf_test.coef_)

print('\n', clf_test.predict(X)[:10])

# 查看结果概率
print(clf_test.predict_proba(X)[:10])

# 进行准确率计算
from sklearn.metrics import accuracy_score
print(accuracy_score(y, clf_test.predict(X)))


# 构建机器学习流
# 将评估器类进行串联形成机器学习流，而不能串联实用函数
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print('\n\n', pipe.fit(X_train, y_train))

# 该过程就相当于两个评估器都进行了训练，然后我们即可使用predict方法，
# 利用pipe对数据集进行预测，当然实际过程是先（借助训练数据的统计量）进行归一化，
# 然后再进行逻辑回归模型预测.
print('\n', pipe.predict(X_test))
print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))

# sklearn的模型保存
# 即可借助joblib包来进行sklearn的模型存储和读取，
# 相关功能非常简单，我们可以使用dump函数进行模型保存，
# 使用load函数进行模型读取
import joblib 
joblib.dump(pipe, 'pipe.model')
pipe1 = joblib.load('pipe.model')
print(pipe1.score(X_train, y_train))
