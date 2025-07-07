 # -*- coding: utf-8 -*-
# 基础数据科学运算库
import numpy as np
import pandas as pd

# 可视化库
import seaborn as sns
import matplotlib.pyplot as plt

# 时间模块
import time

# sklearn库
# 数据预处理
from sklearn import preprocessing
 
from sklearn.compose import ColumnTransformer

# 实用函数
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 常用评估器
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 网格搜索
from sklearn.model_selection import GridSearchCV

# 自定义评估器支持模块
from sklearn.base import BaseEstimator, TransformerMixin

# 自定义模块
from telcoFunc import *

# re模块相关
import inspect, re


# Part 3.特征衍生与特征筛选
# 接下来导入数据并执行Part 1中的数据清洗步骤。

# 读取数据
tcc = pd.read_csv(r'D:\pythonPractice\practice_example\TelcoCustomerChurn\WA_Fn-UseC_-Telco-Customer-Churn.csv')

def features_test(new_features,
                  features, 
                  labels , 
                  category_cols, 
                  numeric_cols):
    """
    新特征测试函数
    
    :param features: 数据集特征
    :param labels: 数据集标签
    :param new_features: 新增特征
    :param category_cols: 离散列名称
    :param numeric_cols: 连续列名称
    :return: result_df评估指标
    """
    
    # 数据准备
    if type(new_features) == np.ndarray:
        name = 'new_features'
        new_features = pd.Series(new_features, name=name)
    # print(new_features)
    
    features = features.copy()
    category_cols = category_cols.copy()
    numeric_cols = numeric_cols.copy()

    features = pd.concat([features, new_features], axis=1)
    # print(features.columns)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=21)
    
    # 划分连续变量/离散变量
    if type(new_features) == pd.DataFrame:
        for col in new_features:
            if new_features[col].nunique() >= 15:
                numeric_cols.append(col)
            else:
                category_cols.append(col)
    
    else:
        if new_features.nunique() >= 15:
            numeric_cols.append(name)
        else:
            category_cols.append(name)

        
    # print(category_cols)
    # 检验列是否划分完全
    assert len(category_cols) + len(numeric_cols) == X_train.shape[1]

    # 设置转化器流
    logistic_pre = ColumnTransformer([
        ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols), 
        ('num', 'passthrough', numeric_cols)
    ])

    num_pre = ['passthrough', preprocessing.StandardScaler(), preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')]

    # 实例化逻辑回归评估器
    logistic_model = logit_threshold(max_iter=int(1e8))

    # 设置机器学习流
    logistic_pipe = make_pipeline(logistic_pre, logistic_model)

    # 设置超参数空间
    logistic_param = [
        {'columntransformer__num':num_pre, 'logit_threshold__penalty': ['l1'], 'logit_threshold__C': np.arange(0.1, 1.1, 0.1).tolist(), 'logit_threshold__solver': ['saga']}, 
        {'columntransformer__num':num_pre, 'logit_threshold__penalty': ['l2'], 'logit_threshold__C': np.arange(0.1, 1.1, 0.1).tolist(), 'logit_threshold__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']}, 
    ]

    # 实例化网格搜索评估器
    logistic_search = GridSearchCV(estimator = logistic_pipe,
                                   param_grid = logistic_param,
                                   scoring='accuracy',
                                   n_jobs = 12)

    # 输出时间
    s = time.time()
    logistic_search.fit(X_train, y_train)
    print(time.time()-s, "s")

    # 计算预测结果
    return(logistic_search.best_score_,
           logistic_search.best_params_,
           result_df(logistic_search.best_estimator_, X_train, y_train, X_test, y_test))
# 标注连续/离散字段
# 离散字段
category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod']

# 连续字段
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 标签
target = 'Churn'

# ID列
ID_col = 'customerID'

# 验证是否划分能完全
assert len(category_cols) + len(numeric_cols) + 2 == tcc.shape[1]

# 连续字段转化
tcc['TotalCharges']= tcc['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)

# 缺失值填补
tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)

# 标签值手动转化 
tcc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
tcc['Churn'].replace(to_replace='No',  value=0, inplace=True)


# 分出标签和特征
features = tcc.drop(columns=[ID_col, target]).copy()
labels = tcc['Churn'].copy()


# 特征衍生基本概念与分类
# 特征衍生，指的是通过既有数据进行新特征的创建 
# 其一是依据数据集特征进行新特征的创建，此时的特征衍生其实是一类无监督的特征衍生，
# 例如把月度费用（'MonthlyCharges'）和总费用（'TotalCharges'）两列相加，创建新的一列；

# 在大多数时候特征衍生特指无监督特征衍生，而有监督的特征衍生我们会称其为目标编码。

# 无论是特征衍生还是目标编码，实现的途径都可以分为两种，
# 其一是通过深入的数据背景和业务背景分析，进行人工字段合成，
# 这种方法创建的字段往往具有较强的业务背景与可解释性，同时也会更加精准、有效的提升模型效果，
# 但缺点是效率较慢，需要人工进行分析和筛选，

# 其二则是抛开业务背景，直接通过一些简单暴力的工程化手段批量创建特征，然后从海量特征池中挑选有用的特征带入进行建模，
# 这种方法简单高效，但工程化方法同质化严重，在竞赛时虽是必备手段，但却难以和其他同样采用工程化手段批量创建特征的竞争者拉开差距。
# 因此，在实际应用时，往往是先通过工程化方法批量创建特征提升模型效果，
# 然后再围绕当前建模需求具体问题具体分析，尝试人工创建一些字段来进一步提升模型效果。

# 基于业务背景的特征创建
# 分析思路:
# 很多业务经验无法被量化，但是却可以作为我们进行特征创建的突破口。
# 例如，用户流失其实是一种很常见的业务场景，
# 一般来说影响用户粘性的因素可能包括服务体验、用户习惯、群体偏好、用户注册时长、同质化竞品等等因素，
# 据此，我们可以在当前数据集中新增两个字段来衡量用户粘性，
# 其一是新人用户标识（专门标记最近1-2个月内入网用户）、其二则是用户购买服务数量。

# 新人用户标识
# 如果用户注册时间较短，则对产品粘性相对较弱，在数据集中tenure字段是描述用户入网时长的字段，而在该字段的所有取值中，考虑到最短续费周期是一个月，因此有一类用户需要重点关注，那就是最近1-2个月内入网用户：这些用户不仅入网时间短，而且在付费周期的规定下，该类用户极有可能在短暂的体验产品服务后在下个月离网，因此我们不妨单独创建一个字段，用来标注tenure字段取值为1的用户。

# 购买服务数量
# 我们还可以计算用户购买的服务数量，一种简单的判断是用户购买的服务越多、用户粘性越大、用户流失的概率越小，
#  我们可以通过简单汇总每位用户所购买的包括增值服务在内的所有服务类别总数作为新的字段，以此来衡量用户粘性。

# new_customer特征创建
# 此处创建一个new_customer字段用于表示是否是新人客户，我们将所有tenure取值为1的用户划分为新人用户，该字段是0/1二分类字段，1表示是新人用户。字段创建过程如下。
# 筛选条件
new_customer_con = (tcc['tenure'] == 1)
# 然后创建该字段
new_customer = new_customer_con.astype(int).values

# 当然，该字段的加入能否提升模型效果，我们可以简单通过计算该字段与标签之间的相关性来进行检验，
# 如果该字段与标签相关性较强，则大概率加入该字段后模型效果会有所提升，相关性检验可以通过如下方式完成：
# 提取数据集标签
y = tcc['Churn'].values


print(np.corrcoef(new_customer, y))
# 能够发现，新人字段和标签呈现正相关，即新人用户流失概率较大，并且0.24的相关系数在所有相关系数计算结果中属于较大值

# new_customer新增字段效果

# 添加new_customer列
features['new_customer'] = new_customer.reshape(-1, 1)

# 数据准备
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=21)

# 检验列是否划分完全
category_new = category_cols + ['new_customer']
assert len(category_new) + len(numeric_cols) == X_train.shape[1]

# 设置转化器流
logistic_pre = ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_new), 
    ('num', 'passthrough', numeric_cols)
])

num_pre = ['passthrough', preprocessing.StandardScaler(), 
           preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans', subsample=200000)]

# 实例化逻辑回归评估器
logistic_model = logit_threshold(max_iter=int(1e8))

# 设置机器学习流
logistic_pipe = make_pipeline(logistic_pre, logistic_model)

# 设置超参数空间
logistic_param = [
    {'columntransformer__num':num_pre, 'logit_threshold__penalty': ['l1'], 'logit_threshold__C': np.arange(0.1, 1.1, 0.1).tolist(), 'logit_threshold__solver': ['saga']}, 
    {'columntransformer__num':num_pre,  'logit_threshold__penalty': ['l2'], 'logit_threshold__C': np.arange(0.1, 1.1, 0.1).tolist(), 'logit_threshold__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']}, 
]

# 实例化网格搜索评估器
logistic_search = GridSearchCV(estimator = logistic_pipe,
                               param_grid = logistic_param,
                               scoring='accuracy',
                               n_jobs = 12)

# 输出时间
s = time.time()
logistic_search.fit(X_train, y_train)
print(time.time()-s, "s")

# 计算预测结果
result_df(logistic_search.best_estimator_, X_train, y_train, X_test, y_test)

print(logistic_search.best_score_)

# 该结果也验证了该特征创建的有效性。同时相比训练集单次运行的准确率结果，我们发现网格搜索中验证集的准确率均值更能代表模型当前泛化能力

# 模型泛化能力评估
# 交叉验证时验证集的平均得分就是这么一个基于后验的过程得到的最有力的模型泛化能力的评估指标，
# 并且在大多数时候，验证集的平均得分越高、测试集上的得分也越高，二者大概率保持一致
# 若是在竞赛时，我们应当将全部有标签的数据带入进行建模并配合交叉验证过程，并在根据验证集平均得分进行参数选择或者特征选择。

# service_num字段创建与效果检验
# 接下来进一步创建用于统计用户总共购买服务数量的字段service_num，
# 原始数据集记录了总共九项服务的用户购买情况，我们可以通过如下方式汇总每位用户总共购买的服务数量：
service_num = ((tcc['PhoneService'] == 'Yes') * 1
             + (tcc['MultipleLines'] == 'Yes') * 1
             + (tcc['InternetService'] == 'Yes') * 1
             + (tcc['OnlineSecurity'] == 'Yes') * 1 
             + (tcc['OnlineBackup'] == 'Yes') * 1 
             + (tcc['DeviceProtection'] == 'Yes') * 1 
             + (tcc['TechSupport'] == 'Yes') * 1
             + (tcc['StreamingTV'] == 'Yes') * 1
             + (tcc['StreamingMovies'] == 'Yes') * 1
            ).values


print(service_num)
features_test(new_features = service_num)
#   能够发现，模型在网格搜索中验证集准确率的均值由0.8042提升至0.8047，约提升了0.05%，而测试集上的准确率由0.793658提升到0.797444，有约0.3%的效果提升。


# 组合字段效果
# 此前定义的features_test函数能够处理DataFrame类型数据，我们只需要将新增的两个特征合成一个新的DataFrame然后带入函数即可。

new_features = pd.DataFrame({'new_customer': new_customer, 'service_num': service_num})
new_features[:5]
features_test(new_features = new_features)
# 在大多数情况下，特征的叠加效果都不会超过单独特征提升效果之和，而当新增特征过多时，还有可能造成维度灾难，从而导致模型效果下降，
# 外加带入的数据过多也会导致计算时长增加，因此当创建了许多特征后，需要谨慎筛选带入建模的特征。

# 基于数据分布规律的字段创建
# 如果说基于业务经验的字段创建是一种基于通识经验的特征衍生方法，
# 那么基于当前数据分布的字段创建方法，则是更加贴合当前实际情况、基于更深度分析然后再进行的特征创建。

# 在大多数情况下，基于数据具体情况进行的特征衍生也会更加有效

# 很多时候特征衍生工作并不是为了创造更多的信息，而是更好的去呈现既有信息，
# 那么一般来说，如果我们创建的特征对应的不同类别比例差异越大、
# 则该特征就约有利于帮助模型完成训练（例如创建的某特征取值为1时对应数据标签全部为1或者全部为0）。
# 们需要从那些本身对标签有较高区分度的特征（如流失率特别大或流失率特别小的特征）入手进行分析



# 人口统计信息字段探索与特征衍生
# 不难发现，在所有三个字段中，老年用户字段'SeniorCitizen'是一个区分度极为明显的字段，经过简单统计分析不难发现，有近半数的老年用户都流失了：

tcc[(tcc['SeniorCitizen'] == 1)]['Churn'].mean()

# 如此少量的人群却有如此高的流失率，我们不妨先从老年字段入手进行分析。
# 当然，一方面，从中我们能够看出该电信运营商提供的产品确实对老年用户不够友好，
# 同时该字段如此高的流失率，也成了我们进一步进行数据探索的突破口。
# 一般来说有响应率较高的字段或者数值异常字段，都可以成为进一步数据探索的突破口。

# 此处我们重点关注的是，在其他关联字段（即用户人口统计信息字段）中，是否有其他某字段对老年用户流失有重要影响，
# 如果有的话，二者字段的组合就能够对用户是否流失进行较好的标记。
# 也就是说，我们这里是希望从老年字段入手，通过和其他关联字段的组合，创建一个新的有效字段。


# 与老年字段同属人口统计信息的字段还有'Dependents'、'Partner'和'gender'三个，并且每个字段都有两个取值水平。我们首先将老年人用户信息单独提取出来：
ts = tcc[tcc['SeniorCitizen']==1]
ts.head(2)

# 后和Part 1类似，通过堆叠柱状图来观察其他不同变量的不同取值对用户是否流失的影响情况：
sns.set()
plt.figure(figsize=(16,12), dpi=200)

plt.subplot(221)
sns.countplot(x="Dependents",hue="Churn",data=ts,palette="Blues", dodge=False)
plt.xlabel("Dependents")
plt.title("Churn by Dependents")

plt.subplot(222)
sns.countplot(x="Partner",hue="Churn",data=ts,palette="Blues", dodge=False)
plt.xlabel("Partner")
plt.title("Churn by Partner")

plt.subplot(223)
sns.countplot(x="gender",hue="Churn",data=ts,palette="Blues", dodge=False)
plt.xlabel("gender")
plt.title("Churn by gender")

# 性别字段对于老年用户来说对流失率的交叉影响并不明显
# Dependents和Partner字段对老年用户人群交叉影响较为明显
# 即流失率在该两个字段的不同取值上差异较大

# 当然我们重点关注那些流失率较大的交叉影响组合结果，即两个字段取值为No的时候人群基本情况
(ts[ts['Partner'] == 'No']['Churn'].mean(),
ts[ts['Partner'] == 'No']['Churn'].shape[0])

(ts[ts['Dependents'] == 'No']['Churn'].mean(),
ts[ts['Dependents'] == 'No']['Churn'].shape[0])

# 假设A类用户为老年且没有伴侣的用户，B类用户为老年且经济不独立的用户，
# 则根据上述统计结果我们发现，两类用户都比单独老年用户流失比例更高
# 这说明如果我们提取这两类人群的划分能够帮我们更好的标识风险人群，
# 我们可以据此衍生出两个不同的字段，即老年且没有伴侣标识字段（字段A）和老年且不经济独立标识字段（字段B）。
# 同时我们也发现，尽管A类用户比B类用户流失率高5%，A类用户人数只有B类用户的一半，
# 这说明A类人群划分尽管有效，但不一定有很好的普适性，对应字段A对模型建模帮助或许不如字段B。

# WOE计算与IV值检验
# 最通用同时也是被广泛验证较为有效的检验一个变量预测能力的方式，就是计算IV(information value)值，
# 此处IV值和决策树的IV值同名，但这里的IV并不是依据C4.5中信息熵计算得来，
# 而是一种简单基于样本比例的计算过程，其基本公式如下：

# 首先，IV值的计算结果是二分类问题中某离散变量的信息价值，或者说该变量对标签取值的影响程度，
# IV值越大说明该字段对标签取值影响越大

PG1 = tcc[tcc['SeniorCitizen'] == 1]['Churn'].value_counts()[1] / tcc['Churn'].value_counts()[1]
PB1 = tcc[tcc['SeniorCitizen'] == 1]['Churn'].value_counts()[0] / tcc['Churn'].value_counts()[0]

IV_1 = (PG1-PB1) * np.log(PG1/PB1)

# 类似的，我们可以继续计算IV0，此时i=0表示'SeniorCitizen'取值为1时子数据集的计算结果:
PG0 = tcc[tcc['SeniorCitizen'] == 0]['Churn'].value_counts()[1] / tcc['Churn'].value_counts()[1]
PB0 = tcc[tcc['SeniorCitizen'] == 0]['Churn'].value_counts()[0] / tcc['Churn'].value_counts()[0]
IV_0 = (PG0-PB0) * np.log(PG0/PB0)

# 最终'SeniorCitizen'这列的IV值为：
print(IV_0 + IV_1)
# 若是需要通过IV值来判断新字段是否有用，则不能简单看新字段的IV值，而是需要用新字段的IV值和原始字段进行对比，新字段IV值至少要比原始字段IV最小值要大，新字段才是有效字段。

# 除了能够作为特征重要性评估手段外，IV值和WOE计算过程也经常用于连续字段分箱中，尤其常见于评分卡模型。


def IV(new_features, DataFrame=tcc, target=target):
    count_result = DataFrame[target].value_counts().values
    
    def IV_cal(features_name, target, df_temp):
        IV_l = []
        for i in features_name:
            IV_temp_l = []
            for values in df_temp[i].unique():
                data_temp = df_temp[df_temp[i] == values][target]
                PB, PG = data_temp.value_counts().values / count_result
                IV_temp = (PG-PB) * np.log(PG/PB)
                IV_temp_l.append(IV_temp)
            IV_l.append(np.array(IV_temp_l).sum())
        return(IV_l)
            
    if type(new_features) == np.ndarray:
        features_name = ['new_features']
        new_features = pd.Series(new_features, name=features_name[0])
    elif type(new_features) == pd.Series:
        features_name = [new_features.name]
    else:
        features_name = new_features.columns

    df_temp = pd.concat([new_features, DataFrame], axis=1)
    df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
    IV_l = IV_cal(features_name=features_name, target=target, df_temp=df_temp)

    res = pd.DataFrame(IV_l, columns=['IV'], index=features_name)
    return(res)

IV(tcc[['SeniorCitizen', 'Partner', 'Dependents']])

custmer_A = (((tcc['SeniorCitizen'] == 1) & (tcc['Partner'] == 'No')) * 1).values
custmer_B = (((tcc['SeniorCitizen'] == 1) & (tcc['Dependents'] == 'No')) * 1).values

new_features = pd.DataFrame({'custmer_A':custmer_A, 'custmer_B':custmer_B})

IV(new_features)
#   能够发现，用于创建A字段的两个原始字段的IV值（0.105621、0.118729）都比A字段IV值要高（0.099502），
# 而B字段的IV值（0.114785）则要高于构建该字段的SeniorCitizen字段的IV值（0.105621），据此我们判断B字段是可用的有效字段。



# 效果检验

# 接下来我们将上述两个字段分别带入模型测试实际效果：
features_test(new_features = custmer_A)
features_test(new_features = custmer_B)

#能够发现，相比原始结果（best_score=0.8042，测试集准确率为0.793658），
# 字段B能够有效提升模型效果，而字段A则帮助不大，据此我们也能够看出IV值在帮助进行特征筛选时的有效性。



# 合约周期字段探索与特征衍生
# 根据Part 1中的数据探索结果，不难发现月付用户流失率极高：
# 我们就从月付费用户入手进行分析，同样，我们先将所有月付费用户单独提取出来：

cm = tcc[tcc['Contract'] == 'month-to-month']

# 观察其他用户账户字段字段在不同取值时标签的分布情况：
sns.set()
plt.figure(figsize=(16,8), dpi=200)

plt.subplot(121)
sns.countplot(x="PaymentMethod",hue="Churn",data=cm,palette="Blues", dodge=False)
plt.xlabel("PaymentMethod")
plt.title("Churn by PaymentMethod")

plt.subplot(122)
sns.countplot(x="PaperlessBilling",hue="Churn",data=cm,palette="Blues", dodge=False)
plt.xlabel("PaperlessBilling")
plt.title("Churn by PaperlessBilling")

# 根据图形结果初步判断按月付费同时是通过电子渠道付款的用户，流失率超过半成，当然我们还需要进一步进行计算：

(cm[cm['PaymentMethod'] == 'Electronic check']['Churn'].mean(),
cm[cm['PaymentMethod'] == 'Electronic check']['Churn'].shape[0])

(cm[cm['PaperlessBilling'] == 'Yes']['Churn'].mean(),
cm[cm['PaperlessBilling'] == 'Yes']['Churn'].shape[0])

# 似的情况又出现了，按月付费的用户的用户中，电子付费的用户流失率高、但人数较少，而无纸化计费（电子合约）的用户更多、但流失率相对低一些
# 而根据上述结果，我们可以创建两个复合字段，分别是按月付费且通过电子渠道付费账户（A），以及按月付费且无纸质合约类账户（B），
# 这两个字段是我们根据上述可视化结果判断最有可能提升模型效果的字段（标签分布与原始标签分布差异最大）

account_A = (((tcc['Contract'] == 'Month-to-month') & (tcc['PaymentMethod'] == 'Electronic check')) * 1).values
account_B = (((tcc['Contract'] == 'Month-to-month') & (tcc['PaperlessBilling'] == 'Yes')) * 1).values

# 原始字段IV值
IV(tcc[['Contract', 'PaymentMethod', 'PaperlessBilling']])
# 新字段IV值
new_features = pd.DataFrame({'account_A':account_A, 'account_B':account_B})
IV(new_features)


# 接下来我们分别将两个字段带入模型进行检验，结果如下：
features_test(new_features = account_A)
features_test(new_features = account_B)

# 总的来说，相比基于业务经验的字段创建过程，基于数据探索的字段创建过程更加“有迹可循”