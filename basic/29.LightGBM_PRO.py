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
from lightgbm import LGBMClassifier,LGBMRegressor

# LGBM的原生API调用
# 流程如下：：
#Step1.Data Interface：借助.Dataset方式进行数据集封装；
#Step2.Setting Parameters：创建超参数字典，用于向模型传输超参数。若完全使用默认参数，则可设置空的字典作作为超参数列表对象；
#Step3.Training：通过.train的方式进行模型训练。
# 读取数据
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2)
print(X_train.head())
print(y_train)
# 数据集创建
train_data = lgb.Dataset(X_train, label=y_train)

# 设置超参数字典
# 这里我们若全部采用模型默认的超参数，只需要设置空字典即可：
param = {}
# 不过这里因为鸢尾花数据是多分类问题，LGBM模型在默认情况下是回归类模型，
# 因此需要通过超参数字典传输建模类型，即objective超参数取值为multiclass（目标为解决多分类问题），
# 同时设置num_class取值为3，即3个类别的多分类问题。

param['objective'] = ['multiclass']
param['num_class'] = [3]

# 进行训练
# 极简流程只需要设置输入数据和超参数字典，其实也是因为lgb.train函数只有这两个必选的参数。
bst = lgb.train(param, train_data)

# 预测
print(bst.predict(X_test))
# 然后即可将概率预测结果进一步转化为类别预测结果，这里同样可以使用argmax进行计算：
print(bst.predict(X_test).argmax(1)) # 返回类别中最大概率的索引，row为每个sample，column为每个类的概率


# 原生API进阶使用
# Data Structure API，即数据和模型构建类AP
# 在LGBM的API设计中，首字母大写的都是类（类似于sklearn的评估器），首字母小写的则都是函数

# 各类的基本解释如下
# Dataset：基础数据集创建类，用于创建LGBM模型训练的数据集；
# Booster：基础模型类，用于（实例化）生成一系列可以用于训练的模型并进行建模后的评估
# CVBooster：基础模型类，和Booster类似，只不过CVBooster实例化的模型支持多次（带入不同数据集）进行训练，并保存多组模型训练结果，方便手动进行交叉验证；
# Sequence：序列数据集创建类，用于创建序列型（如时间序列或者排序序列）数据集

# Dataset类的解释与使用方法
# 便捷读取和存储：不仅可以读取当前编程环境中的Numpy和Pandas对象，同时能够直接读取本地文件，并支持多种文件格式类型的读取
# 更多功能拓展：能够在读取数据文件时标注离散变量，以及对离散变量自动编码、自动设置数据集各样本权重等；
# Dataset数据格式能够显著降低内存，同时提高LGBM算法的计算效率
# 支持分布式计算和GPU加速计算


# 通用的查看原始完整数据集的流程是创建Dataset对象时设置free_raw_data=False，
# 然后训练一次模型（相当于加载数据），然后再通过各种方法查看原始数据集。
#若设置了free_raw_data=False，则这个临时数据集就会变成长期存在的数据集，可以通过.get_等一系列方法来溯源获得
# ，而在默认参数free_raw_data=True的情况下，这个创建的临时数据集将在加载完（也就是模型训练完）之后被释放掉，以此来进行有效的内存管理。
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col=0)
train_data = lgb.Dataset(data=data.iloc[:, 1:-1], label=data.iloc[:, -1], free_raw_data=False)

# 在加载数据集之前是无法查看数据集和特征名，只能查看标签信息：
print(train_data.get_label())

# 加载数据集，用一次模型训练即可完成数据集加载：
param = {}
bst = lgb.train(param, train_data)
# 此时即可查看数据集原始数据和特征名称：
print(train_data.get_data())
print(train_data.get_feature_name())


# 借助.construct()方法进行数据集加载
# 也可以考虑使用.construct()方法进行数据加载，提前验证数据集正确性：
train_data = lgb.Dataset(data=data.iloc[:, 1:-1], label=data.iloc[:, -1], free_raw_data=False)
train_data.construct()
print(train_data.get_data())
# 而在工业实践中，更为通用的做法是特征工程阶段和模型训练阶段相对独立，
# 在执行完特征工程后，将这些已经处理好的特征进行本地文件保存，
# 然后使用Dataset对本地文件直接进行读取。

# LibSVM (zero-based)是LibSVM（是一种用于支持向量机SVM训练的软件包）的最常用数据格式，
# 这种文本文件格式使用空格或制表符分隔特征和标签，并使用稀疏表示法来存储特征值，可以有效地压缩数据


data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col=0)
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# 数据处理好了之后即可进行本地保存。
# 若要LGBM直接读取本地文件，则在保存过程中需要注意两点，
# 其一是LGBM只支持utf-8的编码格式的本地文本文件读取
# 其二则是当前LGBM对列名称的读取存在一定的“障碍”，
# 即无法顺利的读取列名称，因此这里建议设置header=None

X_train.to_csv('X_train.csv', index=False, header=None, encoding='utf-8')
X_test.to_csv('X_test.csv', index=False, header=None, encoding='utf-8')
y_train.to_csv('y_train.csv', index=False, header=None, encoding='utf-8')
y_test.to_csv('y_test.csv', index=False, header=None, encoding='utf-8')

# 此时保存的本地文件列名称是原始数据集的第一行：
data_temp = pd.read_csv('X_train.csv')
print(data_temp.head())

# 这里我们也可以在读取的时候设置header=None以消除该问题：
data_temp = pd.read_csv('X_train.csv', header=None)
data_temp.head()

# 在LGBM的Dataset函数读取文件的过程中，会自动将csv的列名称一行识别为第一行数据，因此lgb.Dataset在进行读取时无需设置，直接读取即可：
train_data = lgb.Dataset('X_train.csv', label=y_train)


# 更常见的做法是保存和读取LGBM binary文件，而非csv文件，用于后续的LGBM建模。
# LGBM binary文件是LGBM专用的一种非常特殊的二进制文件，可以借助二进制制式进行高效快速的数据存储，并被LGBM模型正确的识别

train_data.save_binary('train.bin')

# 然后在读取时，直接使用lgb.Dataset即可进行读取：
# 此时只需要在Dataset参数位置上输入一个二进制文件的读取地址，
# 就能完整创建一个包含全部train_data信息的Dataset对象。
# 需要注意的是，这里不需要额外输入标签，"train.bin"文件本身就包含标签信息
train_data_bin = lgb.Dataset('train.bin')
param = {}
bst = lgb.train(param, train_data_bin)

print(train_data_bin.num_feature())
print(train_data_bin.get_label())


# 借助Dataset标记离散变量
# 通过输入categorical_feature参数来进行离散变量的标记：
# 我们可以在创建Dataset对象时输入categorical_feature=cate_features，来标记这些离散列：







#  LGBM高效数据保存和读取流程
# 在熟悉了一系列LGBM在保存和读取数据相关过程后，我们来总结一套通用的、面向LGBM的高效数据保存和读取流程
# Step 1.将本地文件读取为Panda数据类型，并进行数据清洗、特征工程等操作；
# Step 2.分别提取训练集和测试集的特征和标签，并分别进行本地存储，避免出现数据丢失等情况
# Step 3.创建Dataset类型对象，并正确输入特征、标签、离散列特征名称等，非必要情况下一般建议free_raw_data=True；
# Step 4.本地保存训练数据和测试数据的LGBM binary文件，方便下次模型训练时调用
# Step 5.若继续进行模型训练，则可以考虑在当前操作空间中删除原始数据文件对象，以减少内存占用率。

# Step 1.读取数据，划分数据集，进行数据清洗等
ata = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv', index_col=0)
print(data.head())
cate_features = ['住宅类型',
                 '街道路面状况',
                 '巷子路面状况',
                 '住宅形状(大概)',
                 '住宅现状',
                 '水电气',
                 '住宅配置',
                 '住宅视野',
                 '社区',
                 '住宅周边1',
                 '住宅周边2',
                 '适用家庭',
                 '住宅房型',
                 '装修质量',
                 '整体质量',
                 '天花板类型',
                 '天花板材料',
                 '户外装饰1',
                 '户外装饰2',
                 '砖墙类型',
                 '户外材料质量',
                 '户外装修质量',
                 '地下室类型',
                 '地下室质量',
                 '花园外墙',
                 '地下室现状1',
                 '地下室现状2',
                 '地下室建造现状',
                 '暖气类型',
                 '暖气质量',
                 '中央空调',
                 '电力系统',
                 '全卫地下室',
                 '半卫地下室',
                 '全卫及以上',
                 '半卫及以上',
                 '卧室及以上',
                 '厨房及以上',
                 '厨房质量',
                 '总房间量',
                 '住宅性能',
                 '壁炉数量',
                 '壁炉质量',
                 '车库类型',
                 '车库装修现状',
                 '车位数量',
                 '车库质量',
                 '车库现状',
                 '石板路',
                 '木板面积',
                 '泳池质量',
                 '篱笆质量',
                 '其他配置',
                 '销售类型',
                 '销售状态']
#Stap 2.划分训练集和测试集，并进行本地保存，这里划分方式和此前XGB课程中划分方式保一致
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:80], data.iloc[:, 80], test_size=0.3, random_state=1412)
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

# Step 3.创建Dataset类型对象
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cate_features)
test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cate_features)

# 同时进行加载，观察数据集是否正确创建
train_data.construct()
test_data.construct()
# Stap 4.本地保存二进制文件，方便后续调用
# 需要提前删除此前创建的train.bin，save_binary不会覆盖之前的同名文件
train_data.save_binary('train.bin')
test_data.save_binary('test.bin')
# Step 5.删除原始数据对象，清理内存
del(data)
del(X_train, y_train, X_test, y_test)
gc.collect()

#然后即可进行模型训练
params = {}
bst = lgb.train(param, train_data, categorical_feature=cate_features)
# 完成了最为通用的LGBM Dataset设置过程。


# LGBM Training API使用方法
# params：字典类型，用于设置模型参数和训练参数。这些参数包括:
# boosting_type: 提升方法类型（如gbdt、rf、dart、goss）
# objective: 目标函数（如binary、multiclass、regression等）
# metric: 评估指标（如binary_logloss、multi_logloss、rmse等）。
# num_leaves: 树的最大叶子数。
# learning_rate: 学习率。
# feature_fraction: 每次迭代中使用特征的比例。
# bagging_fraction: 每次迭代中使用数据的比例。
# bagging_freq: 控制bagging的频率。
# train_set：训练数据集，必须是Dataset类型的实例。
# num_boost_round：迭代次数，即树的数量。
# valid_sets：验证数据集列表，可以包含多个Dataset对象。用于在训练过程中评估模型性能。
# valid_names：验证数据集的名称列表，用于输出时区分不同的验证数据集。
# init_model：初始化模型，可以是一个已经训练好的模型文件路径或一个Booster对象。用于从已有模型继续训练。
# early_stopping_rounds：早停策略。如果在指定轮数内验证集上的评估指标没有改善，则停止训练。
# learning_rates：可以是一个函数或列表，定义每轮迭代的学习率。


# train函数基本调用流程

# 加载和准备数据
iris = load_iris()
X = iris.data
y = iris.target

# 将标签二值化，仅仅为了二分类任务
y = (y == 0).astype(int)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建LightGBM的数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, y_test, reference=train_data)
# 当训练集和验证集分布相似，且需加速验证过程或保证分箱一致性时，推荐使用 reference=train_data

# 4. 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves' : 31,
    'learning_rate': 0.05,
    'feature_fraction':0.9
}
# 存储评估结果
evals_result = {}

# 训练模型
bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data], valid_names=['valid'],
                early_stopping_rounds=10, evals_result=evals_result, verbose_eval=10)

# 6. 预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = (y_pred > 0.5).astype(int)
print(y_pred)


# 7. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 输出评估结果，绘制图片
print(evals_result)
plt.plot(np.arange(1, 101), evals_result['valid']['binary_logloss'])
plt.show()



# 可视化每轮迭代的损失值
def plot_metric(evals_result, metric_name):
    for dataset in evals_result.keys():
        metric_values = evals_result[dataset][metric_name]
        plt.plot(metric_values, label=f'{dataset} {metric_name}')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over iterations')
    plt.legend()
    plt.show()

# 可视化 binary_logloss
plot_metric(evals_result, 'binary_logloss')




# 回归类问题建模流程
data = pd.read_csv("train_encode.csv",index_col=0)
features = data.iloc[:, :80]
labels = data.iloc[:, 80]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

from sklearn.metrics import mean_squared_error
# 3. 创建LightGBM的数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',  # 使用均方误差作为评估指标
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 5. 存储评估结果
evals_result = {}

# 6. 训练模型
bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[train_data, valid_data], valid_names=['train', 'valid'],
                early_stopping_rounds=10, evals_result=evals_result, verbose_eval=10)

# 7. 预测和评估
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test data: {mse}')


# lgb.cv：自动交叉验证训练方法
# nfold: 折数，指定交叉验证的折数（例如 5 表示 5 折交叉验证）。
# stratified: 布尔值，是否进行分层抽样，主要用于分类任务，确保每个折中类别比例一致。
# shuffle: 布尔值，是否在每次分割前进行数据洗牌
# metrics: 自定义评估指标，可以是一个字符串或列表，指定要监控的评估指标。
# folds: 用户自定义的交叉验证折叠生成器，可以替代 nfold 和 shuffle 参数。
# seed: 随机种子，用于数据分割和特征的随机化。
# eval_train_metric: 布尔值，是否在训练过程中输出训练数据的评估指标。
# return_cvbooster: 布尔值，是否返回交叉验证中的每个模型。

data = pd.read_csv("train_encode.csv",index_col=0)
features = data.iloc[:, :80]
labels = data.iloc[:, 80]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# 3. 创建LightGBM的训练数据集
train_data = lgb.Dataset(X_train, label=y_train)
# 4. 设置参数
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',  # 使用均方误差作为评估指标
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 5. 使用lgb.cv在训练集上进行交叉验证，并记录每轮迭代的训练和验证损失值
cv_results = lgb.cv(params, train_data, num_boost_round=100, nfold=5, stratified=False, shuffle=True,
                    metrics='l2', early_stopping_rounds=10, verbose_eval=10, seed=42, return_cvbooster=True)



# 超参数优化
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
data = pd.read_csv("train_encode.csv",index_col=0)
features = data.iloc[:, :80]
labels = data.iloc[:, 80]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# 创建LightGBM的训练数据集
train_data = lgb.Dataset(X_train, label=y_train)

# 定义参数空间
param_space = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': hp.choice('num_leaves', range(20, 150)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    'bagging_freq': hp.choice('bagging_freq', range(1, 10)),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', range(20, 150)),
    'max_depth': hp.choice('max_depth', range(5, 20)),
    'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
    'lambda_l2': hp.uniform('lambda_l2', 0.0, 1.0),
    'feature_pre_filter': False,  
    'verbose': -1  # 设置为-1来减少输出信息
}

def objective(params):
    # 每次运行时创建新的Dataset对象
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # 使用lgb.cv进行交叉验证
    cv_results = lgb.cv(params, train_data, num_boost_round=200, nfold=5, stratified=False, shuffle=True,
                        metrics='l2', early_stopping_rounds=10, verbose_eval=False, seed=42)
    
    # 获取最小的均方误差
    best_loss = min(cv_results['l2-mean'])
    
    # 返回结果
    return {'loss': best_loss, 'status': STATUS_OK}

# 运行Hyperopt进行优化
trials = Trials()
best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=100, trials=trials)

# 打印最佳参数
print("Best parameters:", best)

# 将最佳参数转换为完整的参数字典
best_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': best['num_leaves'],
    'learning_rate': best['learning_rate'],
    'feature_fraction': best['feature_fraction'],
    'bagging_fraction': best['bagging_fraction'],
    'bagging_freq': best['bagging_freq'],
    'min_data_in_leaf': best['min_data_in_leaf'],
    'max_depth': best['max_depth'],
    'lambda_l1': best['lambda_l1'],
    'lambda_l2': best['lambda_l2'],
    'feature_pre_filter': False,
    'verbose': -1
}

# 重新创建训练数据集
train_data = lgb.Dataset(X_train, label=y_train)

# 训练最终模型
bst = lgb.train(best_params, train_data, num_boost_round=100)

# 在测试集上进行预测
y_test_pred = bst.predict(X_test)

# 计算均方误差
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error on test data: {test_mse}')