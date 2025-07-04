import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Part 2.数据编码与模型训练
tcc = pd.read_csv(r'D:\pythonPractice\practice_example\TelcoCustomerChurn\WA_Fn-UseC_-Telco-Customer-Churn.csv')

#  标注连续/离散字段
# 离散字段
category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod']

# 连续字段
# # 注意，此处我们暂时将tenure划为连续性字段，以防止后续One-Hot编码时候诞生过多特征。
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 标签
target = 'Churn'

# ID列
ID_col = 'customerID'

# 验证是否划分能完全
assert len(category_cols) + len(numeric_cols) + 2 == tcc.shape[1]


# 进行连续变量的缺失值填补：
tcc['TotalCharges']= tcc['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)
tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)
tcc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
tcc['Churn'].replace(to_replace='No',  value=0, inplace=True)

# 当然，清洗完后的数据需要进行进一步重编码后才能带入进行建模，
# 在考虑快速验证不同模型的建模效果时，需要考虑到不同模型对数据编码要求是不同的

# 离散字段的数据重编码
# 我们对离散特征进行哑变量的变换过程其实就是一个数据重编码的过程。

# 不同类型的字段由不同的编码方式，例如文本类型字段可能需要用到CountVector或TF-IDF处理、时序字段可能需要分段字典排序等
# 不同模型对于数据编码类型要求也不一样，例如逻辑回归需要对多分类离散变量进行哑变量变换，
# 而CatBoost则明确要求不能对离散字段字段进行哑变量变换、否则会影响模型速度和效果。

# 本部分我们先介绍较为通用的离散字段的编码方法，然后再根据后续实际模型要求，选择不同编码方式对数据进行处理。

# 1.OrdinalEncoder自然数排序
# 即先对离散字段的不同取值进行排序，然后对其进行自然数值取值转化
from sklearn import preprocessing
# 和所有的sklearn中转化器使用过程类似，需要先训练、后使用：
X1 = np.array([['F'], ['M'], ['M'], ['F']])

# 实例化转化器
enc = preprocessing.OrdinalEncoder()

# 训练
enc.fit(X1)

# 对数据进行转化
print(enc.transform(X1))

# 当我们训练好了一个转化器后，接下来我们就能使用该转化器进一步依照该规则对其他数据进行转化
X2 = np.array([['M'], ['F']])
enc.transform(X2)


# OneHotEncoder独热编码
# 独热编码过程其实和我们此前介绍的哑变量创建过程一致（至少在sklearn中并无差别）。
# 对于独热编码的过程，我们可以通过pd.get_dummies函数实现，也可以通过sklearn中OneHotEncoder评估器（转化器）来实现。

enc = preprocessing.OneHotEncoder()
enc.fit_transform(X1).toarray()

# 并能够对新的数据依据原转化规则进行转化：
enc.transform(X2).toarray()

# 对于独热编码的使用，有一点是额外需要注意的，那就是对于二分类离散变量来说，独热编码往往是没有实际作用的
# 在进行独热编码转化的时候会考虑只对多分类离散变量进行转化，而保留二分类离散变量的原始取值。
# 此时就需要将OneHotEncoder中drop参数调整为'if_binary'，以表示跳过二分类离散变量列。

# 该过程就相当于是二分类变量进行自然数编码，对多分类变量进行独热编码。
X3 = pd.DataFrame({'Gender': ['F', 'M', 'M', 'F'], 'Income': ['High', 'Medium', 'High', 'Low']})
print(X3)
drop_enc = preprocessing.OneHotEncoder(drop='if_binary')
print(drop_enc.fit_transform(X3).toarray())

# 对于sklearn的独热编码转化器来说，尽管其使用过程会更加方便，但却无法自动创建转化后的列名称，
# 而在需要考察字段业务背景含义的场景中，必然需要知道每一列的实际名称
# 因此我们需要定义一个函数来批量创建独热编码后新数据集各字段名称的函数。

# 提取原始列名称
cate_cols = X3.columns.tolist()
cate_cols_new = []
# 提取独热编码后所有特征的名称
for i, j in enumerate(cate_cols):
    if len(drop_enc.categories_[i]) == 2:  # 只有两类的情况
        cate_cols_new.append(j)
    else: 
        for f in drop_enc.categories_[i]:  # 多类别
            feature_name = j + '_' + f
            cate_cols_new.append(feature_name)

# 组成新的dataframe
pd.DataFrame(drop_enc.fit_transform(X3).toarray(), columns=cate_cols_new)


def cate_colName(Transformer, category_cols, drop='if_binary'):
    """
    离散字段独热编码后字段名创建函数
    
    :param Transformer: 独热编码转化器
    :param category_cols: 输入转化器的离散变量
    :param drop: 独热编码转化器的drop参数
    """
    
    cate_cols_new = []
    col_value = Transformer.categories_
    
    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') & (len(col_value[i]) == 2):
            cate_cols_new.append(j)
        else:
            for f in col_value[i]:
                feature_name = j + '_' + f
                cate_cols_new.append(feature_name)
    return(cate_cols_new)

# 3.ColumnTransformer转化流水线
# 该评估器和pipeline类似，能够集成多个评估器（转化器），
# 并一次性对输入数据的不同列采用不同处理方法，并输出转化完成并且拼接完成的数据。

from sklearn.compose import ColumnTransformer
# 该参数的基本格式为：
# (评估器名称（自定义）, 转化器, 数据集字段（转化器作用的字段）)
# 例如，如果我们需要对tcc数据集中的离散字段进行多分类独热编码
# ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols)

# ColumnTransformer可以集成多个转化器，即可以在一个转化流水线中说明对所有字段的处理方法。
# 例如上述转化器参数只说明了需要对数据集中所有category_cols字段进行转化，
# 而对于tcc数据集来说，还有numeric_cols，也就是连续性字段
# ('num', 'passthrough', numeric_cols)
# 此处出现的'passthrough'字符串表示直接让连续变量通过，不对其进行任何处理。
# 如果需要对连续变量进行处理，如需要对其进行归一化或者分箱，则将'passthrough'改为对应转化器。

preprocess_col = ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols), 
    ('num', 'passthrough', numeric_cols)
])

# 训练转化器
preprocess_col.fit(tcc)
print(pd.DataFrame(preprocess_col.transform(tcc)))
# 转化后的数据仍然是离散字段排在连续字段前面，和两个转化器集成顺序相同

# 转化后离散变量列名称
category_cols_new = cate_colName(preprocess_col.named_transformers_['cat'], category_cols)
# 所有字段名称
cols_new = category_cols_new + numeric_cols

# 输出最终dataframe
pd.DataFrame(preprocess_col.transform(tcc), columns=cols_new)

# 此外，在使用ColumnTransformer时我们还能自由设置系数矩阵的阈值，
# 通过sparse_threshold参数来进行调整，默认是0.3，即超过30%的数据是0值时，ColumnTransformer输出的特征矩阵是稀疏矩阵



# 二、连续字段的特征变换
# 数据标准化与归一化
# 对连续变量而言，标准化可以消除量纲影响并且加快梯度下降的迭代效率，而归一化则能够对每条数据进行进行范数单位化处理

# sklearn中的归一化其实是分为标准化（Standardization）和归一化（Normalization）两类
# Z-Score标准化和0-1标准化，都属于Standardization的范畴

# 而在sklearn中，Normalization则特指针对单个样本（一行数据）利用其范数进行放缩的过程

# 可以通过实用函数来进行标准化处理，同时也可以利用评估器来执行标准化过程
# Z-Score标准化的评估器实现方法

# 用函数进行标准化处理，尽管从代码实现角度来看清晰易懂，但却不适用于许多实际的机器学习建模场景。
# 其一是因为在进行数据集的训练集和测试集切分后，
# 我们首先要在训练集进行标准化、然后统计训练集上统计均值和方差再对测试集进行标准化处理，
# 因此其实还需要一个统计训练集相关统计量的过程；

# 其二则是因为相比实用函数，sklearn中的评估器其实会有一个非常便捷的串联的功能，
# sklearn中提供了Pipeline工具能够对多个评估器进行串联进而组成一个机器学习流，
# 从而简化模型在重复调用时候所需代码量，因此通过评估器的方法进行数据标准化，其实是一种更加通用的选择。


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = np.arange(15).reshape(5, 3)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X)
print(X_train,'\n',  X_test)

scaler.fit(X_train)

# 此处标准化的评估器的训练结果实际上是对输入数据的相关统计量进行了汇总计算，
# 也就是计算了输入数据的均值、标准差等统计量，后续将用这些统计量对各数据进行标准化计算

# 查看训练数据各列的标准差
scaler.scale_

# 查看训练数据各列的均值
scaler.mean_

# 查看训练数据各列的方差
scaler.var_

# 总共有效的训练数据条数
scaler.n_samples_seen_

# 利用训练集的均值和方差对训练集进行标准化处理 （z-score）
scaler.transform(X_train)

# 利用训练集的均值和方差对测试集进行标准化处理
scaler.transform(X_test)

# 0-1标准化的评估器实现方法
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(X)

print(scaler.data_min_)
print(scaler.data_max_)

# 归一化 Normalization
# sklearn中的归一化特指将单个样本（一行数据）放缩为单位范数（1范数或者2范数为单位范数）的过程
# 而sklearn中的Normalization过程，实际上就是将每一行数据视作一个向量，然后用每一行数据去除以该行数据的1-范数或者2-范数。
print(preprocessing.normalize(X, norm='l1'))

# 2-范数单位化过程
print(preprocessing.normalize(X, norm='l2'))


# 也可以通过调用评估器来实现上述过程：
from sklearn.preprocessing import Normalizer
normlize = Normalizer()  # 默认l2
normlize.fit_transform(X)

normlize = Normalizer(norm='l1')
normlize.fit_transform(X)



# 连续变量分箱
# 在实际模型训练过程中，我们也经常需要对连续型字段进行离散化处理，也就是将连续性字段转化为离散型字段。
# 离散之后字段的含义将发生变化，原始字段Income代表用户真实收入状况，而离散之后的含义就变成了用户收入的等级划分:
# 0表示低收入人群、1表示中等收入人群、2代表高收入人群。
# 连续字段的离散化能够更加简洁清晰的呈现特征信息，并且能够极大程度减少异常值的影响,同时也能够消除特征量纲影响，
# 当然，最重要的一点是，对于很多线性模型来说，连续变量的分箱实际上相当于在线性方程中引入了非线性的因素，从而提升模型表现。
# 当然，连续变量的分箱过程会让连续变量损失一些信息，而对于其他很多模型来说（例如树模型），
# 分箱损失的信息则大概率会影响最终模型效果。


# 一般来说分箱的规则基本可以由业务指标来确定或者由某种计算流程来确定。
# 根据业务指标确定:
# 在一些有明确业务背景的场景中，或许能够找到一些根据长期实践经验积累下来的业务指标来作为划分依据

# 根据计算流程确定
# 常见方法有四种，分别是等宽分箱（等距分箱）、等频分箱（等深分箱）、聚类分箱和有监督分箱

# 等宽分箱
#  需要先确定划分成几分，然后根据连续变量的取值范围划分对应数量的宽度相同的区间，并据此对连续变量进行分箱。
# sklearn的预处理模块中调用KBinsDiscretizer转化器实现该过程：
# 我们可以在n_bins参数位上输入需要分箱的个数，
# strategy参数位上输入等宽分箱、等频分箱还是聚类分箱，
# encode参数位上输入分箱后的离散字段是否需要进一步进行独热编码处理或者自然数编码。
# 转化为列向量
income = np.array([0, 10, 180, 30, 55, 35, 25, 75, 80, 10]).reshape(-1, 1)
dis = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
dis.fit_transform(income)
# 需要主要注意的是，这些分箱的边界也就是模型测试阶段对测试集进行分箱的依据
print(dis.bin_edges_)


# 等频分箱
# 需要先确定划分成几分，然后选择能够让每一份包含样本数量相同的划分方式。

# 两分等频分箱，strategy选择'quantile'
dis = preprocessing.KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')

# 当然，如果样本数量无法整除等频分箱的箱数，则最后一个“箱子”将包含余数样本。例如对10条样本进行三分等频分箱，则会分为3/3/4的结果：

# 从上述等宽分箱和等频分箱的结果不难看出，等宽分箱会一定程度受到异常值的影响，而等频分箱又容易完全忽略异常值信息，
# 从而一定程度上导致特征信息损失，而若要更好的兼顾变量原始数值分布，则可以考虑使用聚类分箱。

# 聚类分箱
# 指的是先对某连续变量进行聚类（往往是KMeans聚类），然后用样本所属类别作为标记代替原始数值，从而完成分箱的过程。
# 此处我们使用KMeans对其进行三类别聚类：
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(income)

# 通过.labels_查看每条样本所属簇的类别：
print(kmeans.labels_)
# 该值也就是离散化后每条样本的取值，该过程将第三条数据单独划分成了一类，
# 这也满足了此前所说的一定程度上保留异常值信息这一要求，
# 能够发现，聚类过程能够更加完整的保留原始数值分布信息。

# KBinsDiscretizer转化器中也集成了利用KMeans进行分箱的过程，只需要在strategy参数中选择'kmeans'即可：
dis = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
dis.fit_transform(income)

# 在实际建模过程中，如无其他特殊要求，建议优先考虑聚类分箱方法。

# 有监督分箱
# 无论是等宽/等频分箱，还是聚类分箱，本质上都是进行无监督的分箱，即在不考虑标签的情况下进行的分箱。
# 而在所有的分箱过程中，还有一类是有监督分箱，即根据标签取值对连续变量进行分箱。在这些方法中，最常用的分箱就是树模型分箱。
# 树模型的分箱有两种，其一是利用决策树模型进行分箱，简单根据决策树的树桩（每一次划分数据集的切分点）来作为连续变量的切分依据，
# 由于决策树的分叉过程总是会选择让整体不纯度降低最快的切分点，因此这些切分点就相当于是最大程度保留了有利于样本分类的信息

# 例如假设y为数据集标签：
y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(income, y)
plt.figure(figsize=(6, 2), dpi=150)
tree.plot_tree(clf)
#plt.show()
# 如果需要对income进行三类分箱的话，则可以选择32.5和65作为切分点，对数据集进行切分
# 这种有监督的分箱的结果其实会极大程度利于有监督模型的构建。
# 但有监督的分箱过程其实也会面临诸如可能泄露数据集标签信息从而造成过拟合、决策树生长过程不稳定、树模型容易过拟合等问题影响。
# 因此，一般来说有监督的分箱可能会在一些特殊场景下采用一些变种的方式来进行


# 上述所介绍的关于连续变量的标准化或分箱等过程，也是可以集成到ColumnTransformer中的。
# 例如，如果同时执行离散字段的多分类独热编码和连续字段的标准化，则可以创建如下转化流：
ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols),
    ('num', preprocessing.StandardScaler(), numeric_cols)
])

# 如果需要同时对离散变量进行多分类独热编码、对连续字段进行基于kmeans的三分箱，则可以创建如下转化流：
ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols),
    ('num', preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans'), numeric_cols)
])


# 逻辑回归模型训练与结果解释
# 此处我们首先考虑构建可解释性较强的逻辑回归与决策树模型，并围绕最终模型输出结果进行结果解读，
# 而在下一节，我们将继续介绍更多集成模型建模过程。

# 设置评估指标与测试集
# 在模型训练开始前，我们需要设置模型结果评估指标。此处由于0：1类样本比例约为3：1，
# 因此可以考虑使用准确率作为模型评估指标，同时参考混淆矩阵评估指标、f1-Score和roc-aux值。

#  需要知道的是，一般在二分类预测问题中，0：1在3：1左右是一个重要界限，
# 若0：1小于3：1，则标签偏态基本可以忽略不计，不需要进行偏态样本处理（处理了也容易过拟合），
# 同时在模型评估指标选取时也可以直接选择“中立”评估指标，如准确率或者roc-auc。

# 而如果0：1大于3：1，则认为标签取值分布存在偏态，
# 需要对其进行处理，如过采样、欠采样、或者模型组合训练、或者样本聚类等，
# 并且如果此时需要重点衡量模型对1类识别能力的话，则更加推荐选择f1-Score。


# 此外，模型训练过程我们也将模仿实际竞赛流程，即在模型训练开始之初就划分出一个确定的、全程不带入建模的测试集
# （竞赛中该数据集标签未知，只能通过在线提交结果后获得对应得分），
# 而后续若要在模型训练阶段验证模型结果，则会额外在训练集中划分验证集。
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

train, test = train_test_split(tcc, test_size=0.3, random_state=21)


# 逻辑回归模型训练
# 对于若要带入离散变量进行逻辑回归建模，则需要对多分类离散变量进行独热编码处理。
# 当然，也是因为我们需要先对数据集进行转化再进行训练，因此我们可以通过创建机器学习流来封装这两步。

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 划分特征和标签
X_train = train.drop(columns=[ID_col, target]).copy()
y_train = train['Churn'].copy()
X_test = test.drop(columns=[ID_col, target]).copy()
y_test = test['Churn'].copy()

# 设置转化器与评估器

# 检验列是否划分完全
assert len(category_cols) + len(numeric_cols) == X_train.shape[1]

# 设置转化器流
logistic_pre = ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols), 
    ('num', 'passthrough', numeric_cols)
])

# 实例化逻辑回归评估器
logistic_model = LogisticRegression(max_iter=int(1e8))

# 设置机器学习流
logistic_pipe = make_pipeline(logistic_pre, logistic_model)

# 模型训练
logistic_pipe.fit(X_train, y_train)

# 查看模型结果
logistic_pipe.score(X_train, y_train)
logistic_pipe.score(X_test, y_test)


# 关于更多评估指标的计算，我们可以通过下述函数来实现，同时计算模型的召回率、精确度、f1-Score和roc-auc值：
def result_df(model, X_train, y_train, X_test, y_test, metrics=
              [accuracy_score, recall_score, precision_score, f1_score, roc_auc_score]):
    res_train = []
    res_test = []
    col_name = []
    for fun in metrics:
        res_train.append(fun(model.predict(X_train), y_train))
        res_test.append(fun(model.predict(X_test), y_test))
        col_name.append(fun.__name__)
    idx_name = ['train_eval', 'test_eval']
    res = pd.DataFrame([res_train, res_test], columns=col_name, index=idx_name)
    return res

print(result_df(logistic_pipe, X_train, y_train, X_test, y_test))

#  一般来说准确率再80%以上的模型就算是可用的模型，但同时也要综合考虑当前数据集情况（建模难度），
# 有些场景（比赛）下80%只是模型调优前的基准线（baseline），而有时候80%的准确率却是当前数据的预测巅峰结果（比赛Top 10）。


# 训练集和测试集的划分方式也会影响当前的输出结果，但建议是一旦划分完训练集和测试集后，就围绕当前建模结果进行优化，
# 而不再考虑通过调整训练集和测试集的划分方式来影响最后输出结果，这么做也是毫无意义的。


# 逻辑回归的超参数调优
# 网格搜索能够帮我们确定一组最优超参数，并且随之附带的交叉验证的过程也能够让训练集上的模型得分更具有说服力。

from sklearn.model_selection import GridSearchCV
# penalty	正则化项
# tol	迭代停止条件：两轮迭代损失值差值小于tol时，停止迭代
# C	经验风险和结构风险在损失函数中的权重

# 对模型结果影响较大的参数主要有两类，其一是正则化项的选择，同时也包括经验风险项的系数与损失求解方法选择，
# 第二类则是迭代限制条件，主要是max_iter和tol两个参数

# 而整个网格搜索过程其实就是一个将所有参数可能的取值一一组合，
# 然后计算每一种组合下模型在给定评估指标下的交叉验证的结果（验证集上的平均值），作为该参数组合的得分，
# 然后通过横向比较（比较不同参数组合的得分），来选定最优参数组合。


# 在默认情况下，搜索的目的是提升模型准确率，但我们也可以对其进行修改，例如希望搜索的结果尽可能提升模型f1-Score，
# 则可在网格搜索实例化过程中调整scoring超参数
# 设置转化器流
logistic_pre = ColumnTransformer([
    ('cat', preprocessing.OneHotEncoder(drop='if_binary'), category_cols), 
    ('num', 'passthrough', numeric_cols)
])

# 实例化逻辑回归评估器
logistic_model = LogisticRegression(max_iter=int(1e8))

# 设置机器学习流
logistic_pipe = make_pipeline(logistic_pre, logistic_model)

# 设置超参数空间
logistic_param = [
    {'logisticregression__penalty': ['l1'], 'logisticregression__C': np.arange(0.1, 2.1, 0.1).tolist(), 'logisticregression__solver': ['saga']}, 
    {'logisticregression__penalty': ['l2'], 'logisticregression__C': np.arange(0.1, 2.1, 0.1).tolist(), 'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']}, 
    {'logisticregression__penalty': ['elasticnet'], 'logisticregression__C': np.arange(0.1, 2.1, 0.1).tolist(), 'logisticregression__l1_ratio': np.arange(0.1, 1.1, 0.1).tolist(), 'logisticregression__solver': ['saga']}
]

# 实例化网格搜索评估器
logistic_search_f1 = GridSearchCV(estimator = logistic_pipe,
                                  param_grid = logistic_param,
                                  scoring='f1',
                                  n_jobs = 12)
import time
s = time.time()
logistic_search_f1.fit(X_train, y_train)
print(time.time()-s, "s")
# 计算预测结果
result_df(logistic_search_f1.best_estimator_, X_train, y_train, X_test, y_test)


# 决策树模型训练与结果解释
# 对于决策树来说，由于并没有类似线性方程的数值解释，因此无需对分类变量进行独热编码转化，直接进行自然数转化即可
# 导入决策树评估器
from sklearn.tree import DecisionTreeClassifier

# 设置转化器流
tree_pre = ColumnTransformer([
    ('cat', preprocessing.OrdinalEncoder(), category_cols),
    ('num', 'passthrough', numeric_cols)
])

# 实例化决策树模型
tree_model = DecisionTreeClassifier()

# 设置机器学习流
tree_pipe = make_pipeline(tree_pre, tree_model)


# 模型训练
tree_pipe.fit(X_train, y_train)

# 计算预测结果
result_df(tree_pipe, X_train, y_train, X_test, y_test)
# 能够发现，模型严重过拟合，即在训练集上表现较好，但在测试集上表现一般。此时可以考虑进行网格搜索，通过交叉验证来降低模型结构风险。

# 在新版sklearn中还加入了ccp_alpha参数，该参数是决策树的结构风险系数，作用和逻辑回归中C的作用类似，但二者取值正好相反（ccp_alpha是结构风险系数，而C是经验风险系数）。
# 此处我们选取max_depth、min_samples_split、min_samples_leaf、max_leaf_nodes和ccp_alpha进行搜索：


tree_pre = ColumnTransformer([
    ('cat', preprocessing.OrdinalEncoder(), category_cols),
    ('num', 'passthorugh', numeric_cols)
])

# 实例化决策树
tree_model = DecisionTreeClassifier()

# 设置机器学习流
tree_pipe = make_pipeline(tree_pre, tree_model)

# 构造包含阈值的参数空间
tree_param = {'decisiontreeclassifier__ccp_alpha': np.arange(0, 1, 0.1).tolist(),
              'decisiontreeclassifier__max_depth': np.arange(2, 8, 1).tolist(), 
              'decisiontreeclassifier__min_samples_split': np.arange(2, 5, 1).tolist(), 
              'decisiontreeclassifier__min_samples_leaf': np.arange(1, 4, 1).tolist(), 
              'decisiontreeclassifier__max_leaf_nodes':np.arange(6,10, 1).tolist()}

# 实例化网格搜索评估器
tree_search = GridSearchCV(estimator = tree_pipe,
                           param_grid = tree_param,
                           n_jobs = 12)
tree_search.fit(X_train, y_train)
print(tree_search.best_score_)
