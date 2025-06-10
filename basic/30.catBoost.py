import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# 数据准备
from sklearn.datasets import load_iris

# CatBoost，全称Categorical Boosting

# 读取数据
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)


# 基础建模流程
# CatBoost并没有区分sklearn API和原生API，
# CatBoost调用过程中核心使用的就是CatBoostClassifier API（分类问题）和
# CatBoostRegressor API（回归问题），
# 这两个主模型API既可以和sklearn API无缝集成，
# 同时也支持一些特定的功能，如GPU加速、
# 以及可以使用自定义数据封装工具pool API进行数据压缩和推理加速等。
clf = CatBoostClassifier()
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))


# 离散特征编码方法：Order-based Target Encoding
# 是CatBoost处理类别特征的一种独特方法，它在保持数据泄漏风险最小化的同时，有效地利用了类别特征的信息。
# 在目标编码中，我们使用类别特征的目标变量统计信息（如均值）来代替类别特征的值。
# Order-based Target Encoding是一种目标编码的变种，
# 它通过对数据进行排序，并在计算过程中引入伪随机数以避免目标泄漏。

# Order-based Target Encoding执行流程
# 排序数据：将数据按照某种顺序排列（如样本索引的顺序或时间戳顺序）。
# 递增计算均值：对于每一个类别，按顺序递增地计算其目标变量的均值。
# 在计算均值时引入一定的噪声，以减少过拟合的风险。

# Order-based Target Encoding编码优势
# 防止目标泄漏：通过有序递增地计算均值，当前样本的目标变量不被用于其自身的编码，这有效地防止了目标泄漏。
# 减少过拟合：通过引入噪声，目标编码变得更加稳健，可以有效减少过拟合的风险。
# 适应类别特征：对于高基数的类别特征，这种方法尤其有效，因为它避免了one-hot编码导致的高维问题。


# CatBoost叶节点权重估计方法
# CatBoost在叶节点权重估计同时融入了GBDT和XGB的叶节点计算公式，并将其命名为Newton 和 Gradient
# 其中 Gradient梯度法就是GBDT的叶节点权重计算公式，即直接使用损失函数的梯度来更新叶节点的权重，
# 在每次迭代中，通过计算损失函数对当前预测值的梯度，将梯度累加到叶节点的权重上，从而更新叶节点的权重
# 而所谓的Newton（牛顿法），则是采用了XGB和LGBM相同的叶节点权重计算公式


# CatBoost分裂增益计算方法
# 在CatBoost中，代表分裂增益的参数为score_function，
# 该参数可选的参数Cosine、L2、NewtonCosine和NewtonL2
# 基本属于DBGT、XGB分裂增益和L2范数的排列组合。


# CatBoost中决策树的生长策略
# 以下是三种主要的树生长策略的详细解释：
# 1.SymmetricTree:
# 是CatBoost的默认树生长策略。这种策略构建对称树，
# 即所有的节点按照相同的规则进行分裂，确保树的结构对称。这种对称性使得树的结构更加规则和可预测。
# 训练和预测速度快。
# 适合大规模数据处理。
# 实现简单，易于并行化。
# 可能不如其他方法灵活，在某些复杂数据集上表现可能稍差。

# Depthwise
# 即每次扩展树的一层，选择当前层的所有节点进行分裂，也就是XGB的生长策略。这种方法通过在每层选择最优的分裂点，逐步增加树的深度。
# 对复杂数据具有更好的表达能力。
# 能够捕捉到更复杂的模式和特征关系。
# 计算和存储成本较高。
# 训练时间可能较长，特别是在大数据集上。

# Lossguide
# 基于损失函数指导的生长方法。与Depthwise方法不同，
# Lossguide策略每次选择对整体损失函数影响最大的节点进行分裂，而不是逐层分裂。
# 也就是LGBM的生长策略。Lossguide特点：
# 对复杂数据具有很强的适应性。
# 能够更有效地减少损失，提高模型性能。
# 计算复杂度高。
# 训练时间可能较长，特别是在大数据集上。

# 模型收缩（Model Shrinkage）是CatBoost用于控制模型复杂度的一种技术，
# 通过逐步减少模型中某些参数的影响，从而防止过拟合并提高模型的泛化能力。
# 在CatBoost中，模型收缩涉及对叶节点权重的调整，
# 通过设置model_shrink_rate和model_shrink_mode参数来实现。

# 模型收缩的目的是通过减少叶节点权重的幅度，使得模型更加平滑，
# 防止模型过于复杂，避免过拟合。
# 这一过程通过对每次迭代新增的叶节点权重进行调整，实现逐步收缩的效果。

# model_shrink_rate 是一个控制模型收缩速率的参数。它表示每次迭代中，叶节点权重将按一定比例进行缩减。
# 默认值：默认为0，表示不进行收缩。
# 作用：通过设置非零的model_shrink_rate，
# 可以使得每次新增的叶节点权重按比例缩减，从而降低模型复杂度。

# model_shrink_mode 是一个控制模型收缩模式的参数。
# 'Constant'：以固定比例进行收缩。
# 'Decreasing'：按逐渐减少的比例进行收缩



# CatBoost模型API调用方法进阶
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/kaixi/Downloads/datasets/House Price/train_encode.csv',index_col=0)
features = data.iloc[:, :80]
labels = data.iloc[:, 80]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# 初始化CatBoost回归器
model = CatBoostRegressor(
    eval_metric='RMSE',
    verbose=0,  # 设置为0减少打印信息
    random_seed=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# 特征重要性评估方法
# 获取特征重要性
feature_importances = model.get_feature_importance()
feature_names = X_train.columns
# 创建数据框
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

cat_features = ['住宅类型',
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
                 '其他配置',
                 '销售类型',
                 '销售状态']


# CatBoost超参数优化
# 网格搜索
from sklearn.model_selection import GridSearchCV
from matplotlib import font_manager
import time

# 创建超参数空间
param_grid = {
    'iterations': [200, 300, 400],
    'depth': [6, 8, 10],
}

# 初始化模型
model = CatBoostRegressor(verbose=0)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(grid_search.best_params_)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(mean_squared_error(y_test, y_pred))



# HyperOPT超参数优化
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# 定义目标函数
def objective(params):
    model = CatBoostRegressor (
        iterations=int(params['iterations']),
        depth=int(params['depth']),
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        border_count=int(params['border_count']),
        verbose=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss':mse, 'status':STATUS_OK}
# XGBoost、LightGBM 等库要求自定义损失函数或评估指标必须返回包含 status 的字典，否则可能抛出异常。

# 自定义搜索空间
space = {
    'iterations': hp.quniform('iterations', 100, 1000, 50),
    'depth': hp.quniform('depth', 4, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'l2_leaf_reg':  hp.uniform('l2_leaf_reg', 1, 10),
    'border_count': hp.quniform('border_count', 32, 128, 16)
}

# 进行超参数优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# 输出最佳参数
print(best)

# 使用最佳模参数进行模型训练
best_model = CatBoostRegressor{
    iterations = int(best['iterations']),
    depth = int(best['depth']),
    learning_rate=best['learning_rate'],
    l2_leaf_reg=best['l2_leaf_reg'],
    border_count=int(best['border_count']),
    vebose=0
}

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)


# CatBoost GPU计算模式
# 初始化CatBoost回归器
model = CatBoostRegressor(
    eval_metric='RMSE',
    verbose=0,  # 设置为0以减少打印信息
    task_type='GPU',  # 在这里设置
    random_seed=42
)


# CatBoost pool API使用方法
# CatBoost的Pool对象是一个用于封装数据的高效数据结构。
# 它将特征数据、标签以及元数据信息（如类别特征、权重等）封装在一起，
# 使得数据处理和模型训练更加高效和简便。
# Pool对象提供了一种结构化的方式来管理数据，
# 特别是在处理包含大量类别特征的数据集时显得尤为重要。

# 创建Pool对象
train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test)

# 初始化CatBoost回归器
model = CatBoostRegressor(
    eval_metric='RMSE',
    verbose=0,  # 设置为0以减少打印信息
    cat_features=cat_features,
    random_seed=42
)

# 训练模型
model.fit(train_pool)

# 预测
y_pred = model.predict(test_pool)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


#CatBoost cv API使用方法
# cv+hyperopt执行高精度超参数搜索
from catboost import cv

# 创建Pool对象
# 创建Pool对象
data_pool = Pool(data=features, label=labels, cat_features=cat_features)
# 定义搜索空间
space = {
    'iterations': hp.quniform('iterations', 100, 1000, 50),
    'depth': hp.quniform('depth', 4, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'border_count': hp.quniform('border_count', 32, 128, 16),
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'GPU',  # 使用GPU进行训练
    'random_seed': 42
}
# 定义目标函数
def objective(params):
    params['iterations'] = int(params['iterations'])
    params['depth'] = int(params['depth'])
    
    cv_results = cv(
        pool=data_pool,
        params=params,
        fold_count=5,  # 5折交叉验证
        shuffle=True,
        partition_random_seed=42,
        verbose=False
    )
    
    # 获取最小的测试集均方误差
    best_loss = np.min(cv_results['test-RMSE-mean'])
    
    return {'loss': best_loss, 'status': STATUS_OK}

# 进行超参数优化
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# 使用最佳参数重新训练模型
best_params['iterations'] = int(best_params['iterations'])
best_params['depth'] = int(best_params['depth'])

model = CatBoostRegressor(**best_params) 
# ** 是 Python 的字典解包操作符，用于将字典中的键值对展开为函数的关键字参数。

# 分割数据集用于训练和测试
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train, cat_features=cat_features)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test data: {mse:.2f}')