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



# 数据清洗步骤。
# 读取数据
tcc = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

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

features = tcc.drop(columns=[ID_col, target]).copy()
labels = tcc['Churn'].copy()

# 导入特征衍生模块
import features_creation as fc
from features_creation import *

# 特征衍生的一般顺序
# Stage 1.时序特征衍生
# 首先，如果数据集中存在时序数据，我们可以优先考虑对时序数据进行特征衍生。
# 时序特征衍生过程并不需要依赖其他任何特征，且衍生出来的特征可以作为备选特征带入到交叉组合或者分组统计的过程中。

# Stage 2.多项式特征衍生
# 紧接着，我们就需要来进行多项式特征衍生了。
# 多项式特征衍生往往只适用于连续变量或者取值水平较多的有序变量，
# 并且在实际操作过程中，需要注意衍生特征取值大小的问题，如果衍生特征的绝对值过大，则需要进行数据标准化处理。
# 这里的数据标准化只需针对演变特征进行处理即可，不需要对原始特征进行处理
# 而具体衍生几阶，一般来说2-3阶都是可以考虑的，如果连续特征较多并且连续特征包括很多有效特征，则可以考虑更高阶的多项式衍生。

# Stage 3.交叉组合特征衍生
# 由于衍生特征的稀疏性会伴随着参与组合的特征数量增加而增加,
# 因此我们会优先考虑两两交叉组合特征衍生，然后再考虑三三组合、甚至考虑四四组合等。
# 由于交叉组合本身是一种非常通用且执行效率非常高的特征衍生方法，
# 因此一般情况下，需要针对所有原始离散变量和部分时序衍生字段（分类水平较少的时序衍生字段）进行两两组合，
# 而是否需要进行三三组合，则需要根据两两组合的情况来决定。

# Stage 4.分组统计特征衍生
# 由于很多时候分组统计特征衍生需要依据交叉组合的结果进行分组，
# 所以分组统计特征衍生一般会放在交叉组合特征衍生之后。
# 同时，多项式的衍生特征也可以作为被分组统计的特征，
# 因此往往在交叉组合和多项式特征衍生之后，才会进行分组统计。

# 最重要的是要找准关键变量（keyCol）。此处关键变量可以是单独的原始变量、也可以是衍生的时序字段、当然也可以双变量（或者多变量）的交叉组合字段。
# 此外，分组统计和目标编码是需要分开的，一般来说我们会在分组统计阶段验证合适的keyCol，而在目标编码阶段直接利用已经挑选好的keyCol进行标签的分组统计。

# Stage 5.NLP特征衍生


# 特征分析
# 在套用特征衍生方法进行特征衍生之前，我们需要简单回顾数据集特征的基本情况
# 我们只需要围绕原始表格进行分析即可。一般分析流程是是先从标签和特殊特征入手进行分析。

# 当前数据集的标签是二分类离散变量，因此可以考虑对其进行离散变量的目标编码
# 同时数据集存在时序特征，尽管是非常粗粒度（也就是精确到月，假设）的时间刻度，但仍然可以进行年、月、季度的周期划分
# 在关键时间点这块，由于原始时间刻度就是距离起止时间的差值，因此我们只需考虑进一步设置一些淡旺季的关键时间点即可

# 根据此前的分析，我们不难发现，原始数据集中存在大量相互关联、共同描述类似事件、并且能够相互互补的字段，
# 如人口统计方面的四个字段
# 这些离散字段都是非常适合采用NLP方法（尤其是TF-IDF）进行特征衍生。

# 数据集分类字段较多，因此适合大规模进行交叉组合，同时这些分类字段本身以及一阶交叉组合出来的字段又可以作为分组依据进一步进行分组统计特征衍生


# 2.时序特征衍生
# 时序特征衍生的特征一方面可以和离散变量进行交叉组合（甚至是二阶组合），
# 另一方面也可以作为后续进行分组统计时的分组变量。
# 对于大多数时序字段，我们都可以直接调用timeSeries函数进行时序特征衍生，
# 但telco数据集的时序特征较为特殊，需要按照此前介绍的方法，手动创建年、月、日和所属季度的特征。

# 进行训练集和测试集的划分：
train, test = train_test_split(tcc, random_state=22)
X_train = train.drop(columns=[ID_col, target]).copy()
X_test = test.drop(columnd=[ID_col, target]).copy()

y_train = train['Churn'].copy()
y_test = test['Churn'].copy()

# 然后围绕tenure列，进行分训练集和测试集的时序特征衍生：
X_train_seq = pd.DataFrame()
X_test_seq = pd.DataFrame()

# 年份衍生
X_train_seq['tenure_year'] = ((72 - X_train['tenure']) // 12) + 2014
X_test_seq['tenure_year'] = ((72 - X_test['tenure']) // 12) + 2014

# 月份衍生
X_train_seq['tenure_month'] = (72 - X_train['tenure']) % 12 + 1
X_test_seq['tenure_month'] = (72 - X_test['tenure']) % 12 + 1

# 独热编码
enc = preprocessing.OneHotEncoder()
enc.fit(X_train_seq)

seq_new = list(X_train_seq.columns)


# 创建带有列名称的独热编码之后的df
X_train_seq = pd.DataFrame(enc.transform(X_train_seq).toarray(), 
                           columns = cate_colName(enc, seq_new, drop=None))

X_test_seq = pd.DataFrame(enc.transform(X_test_seq).toarray(), 
                          columns = cate_colName(enc, seq_new, drop=None))

# 首先进行index调整
X_train_seq.index = X_train.index
X_test_seq.index = X_test.index

# 然后进行数据集拼接
df_temp = pd.concat([X_train_seq, y_train], axis=1)

# 和此前一样，我们可以先通过相关系数，简单验证衍生的时序特征和标签之间的关系
df_corr = df_temp.corr()['Churn'].sort_values(ascending = False)


# 挑选前两个相关系数最高的特征进行测试。这里我们仍然考虑使用features_test函数进行测试
new_col = list(np.abs(df_corr).sort_values(ascending = False)[1: 3].index)

train_new_temp = X_train_seq[new_col]
test_new_temp = X_test_seq[new_col]

features_test(train_new_temp, 
              test_new_temp, 
              X_train, 
              X_test, 
              y_train, 
              y_test, 
              category_cols, 
              numeric_cols)

# 能够发现，单独的时序特征衍生就已经达到了非常好的效果



# 3.多项式特征衍生
# 由于原始数据集较为简单，只有两个连续变量，因此我们只需要考虑这两个变量的多项式计算即可：
colNames = ['MonthlyCharges', 'TotalCharges']

X_train_ply, X_test_ply, colNames_train_new, colNames_test_new = Polynomial_Features(colNames=colNames, 
                                                                                     degree=3,
                                                                                     X_train=X_train, 
                                                                                     X_test=X_test)

# 对其进行数据标准化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_ply)

X_train_ply = pd.DataFrame(scaler.transform(X_train_ply), columns=colNames_train_new)
X_train_ply.index = X_train.index

X_test_ply = pd.DataFrame(scaler.transform(X_test_ply), columns=colNames_test_new)
X_test_ply.index = X_test.index

# 接下来测试衍生特征与标签的相关性：
# 然后进行数据集拼接
df_temp = pd.concat([X_train_ply, y_train], axis=1)
df_corr = df_temp.corr()['Churn'].sort_values(ascending=False)

# 对比原始字段，我们发现衍生字段和标签仍然呈现出了一定的相关关系，但由于并没有出现衍生字段强于原始字段的情况，因此无需考虑进行更高阶的多项式衍生了。接下来带入到模型当中进行测试：
new_col = list(np.abs(df_corr).sort_values(ascending = False)[1: 3].index)
print(new_col)

train_new_temp = X_train_ply[new_col]
test_new_temp = X_test_ply[new_col]

features_test(train_new_temp, 
              test_new_temp, 
              X_train, 
              X_test, 
              y_train, 
              y_test, 
              category_cols, 
              numeric_cols)


# 交叉组合特征衍生
# 需要注意的是，在实际建模过程中，特征衍生和后续环节（如特征筛选和模型验证等）并不是完全分割的，而是交叉进行的，
# 也就是完全可以衍生一部分特征后就进行验证，验证后再进行更深入的特征衍生等等。
# 不过无论是何种情况，在特征衍生的开始，我们都可以一次性的进行“全特征衍生”，
# 也就是带入全部离散变量进行两两交叉组合。


# 原始特征的两两交叉组合
# 查看每个分类变量的取值水平
for feature in tcc[category_cols]:
        print(f'{feature}: {tcc[feature].unique()}')


# 先尝试带入原始变量的离散变量进行双变量的两两交叉特征衍生
features_train_new, features_test_new, colNames_train_new, colNames_test_bew = Cross_combination(category_cols, X_train, X_test)

# 在进行交叉组合特征衍生时，无需对object对象类型进行自然数编码（ordinary encode），
# 我们需要原始object对象类型中的字符串来协助创建衍生列的列名称。

# 首先我们可以借助相关系数来初步评估衍生特征和标签的相关关系。这
# 里我们首先需要修改衍生特征的index，使其与训练集的index一致，方便后续进行标签列的拼接：
features_train_new.index = X_train.index

# 相关系数统计
df_temp = pd.concat([features_train_new, y_train], axis=1)
df_corr = df_temp.corr()['Churn'].sort_values(ascending=False)

np.abs(df_corr).sort_Values(ascending = False)[1:11]

# 在衍生特征中，出现了非常多个和标签相关性较强的特征
# 其原始字段与标签的相关性本来就很强，交叉组合只是进一步增强了这些特点。当然，也有一些原本“默默无闻”的字段，通过交叉组合后相关性明显增加
# 有一些字段如SeniorCitizen，原始状态下呈现出和标签的正相关，经过交叉组合后，衍生字段却大多与标签呈现出负相关。这也一定程度上也说明交叉组合会一定程度丰富特征表现。

# 这也是我们为何要花费大量时间来讨论衍生列取名问题的原因。如果不确定列名称，则无法“溯源”有效的衍生列背后是由哪些原始列构成。


# 模型验证
# 接下来我们考虑相关系数最高的三项特征带入到逻辑回归模型中进行模型验证
new_col = list(np.abs(df_corr).sort_values(ascending = False)[1:4].index)

train_new_temp = features_train_new[new_col]
teset_new_temp = features_test_new[new_col]

features_test(train_new_temp, 
              test_new_temp, 
              X_train, 
              X_test, 
              y_train, 
              y_test, 
              category_cols, 
              numeric_cols)

# 能够看出，模型建模结果也达到了非常高的水平


# 原始特征的多变量交叉组合
# 一些原本就很强的特征经过交叉组合后表现出了更强的相关性，
# 这不禁让我们想要进一步尝试围绕这些强相关的特征进行更进一步进行多变量交叉组合特征衍生
# 组合后效果最好的三个特征，就是'OnlineSecurity'、'Contract'和'TechSupport'：
colNames = ['OnlineSecurity', 'Contract', 'TechSupport']
features_train_new, features_test_new, colNames_train_new, colNames_test_new = Cross_Combination(colNames, 
                                                                                                 X_train, 
                                                                                                 X_test, 
                                                                                                 multi=True)

# 当参与交叉组合的特征越多、取值水平越多，衍生特征中出现零值的比例也就越高。
# 在大多数情况下，0值占比较高列有效信息也较少，在特征筛选环节中往往也是要被剔除的。这里能够发现，几乎所有衍生列的0值占比都超过了90%
#应证了此前所说，伴随着参与交叉组合的特征数量增加，有效信息会迅速衰减的过程。
# 有种方法能够降低交叉组合衍生特征矩阵的稀疏性——降低原始离散变量的取值水平。
features_train_new.index = X_train.index
df_temp = pd.concat([features_train_new, y_train], axis=1)
new_col = list(np.abs(df_corr).sort_values(ascending = False)[1: 2].index)
train_new_temp = features_train_new[new_col]
test_new_temp = features_test_new[new_col]

features_test(train_new_temp, 
              test_new_temp, 
              X_train, 
              X_test, 
              y_train, 
              y_test, 
              category_cols, 
              numeric_cols)

# 整体来看，伴随着参与交叉组合的特征越多，有效信息衰减的越严重，因此在大多数时候，我们在执行多变量交叉组合的过程中，都是优先考虑最强特征的组合。


# 带入时序衍生特征的交叉组合衍生
# 首先需要注意的是，对于交叉组合特征衍生只适用于取值水平较少的离散变量（否则衍生特征矩阵会过于稀疏），
# 因此对于时序衍生特征来说，一般只考虑带入年份（如果取值较少的话）、季度、星期几，
# 最多可以考虑带入月份（12个取值的分类变量）。

# 双变量交叉组合特征衍生
# 首先是时序特征和原始特征的两两交叉组合。首先我们将原始数据集中的离散变量与时序衍生特征进行拼接：

X_train_seq.index = X_train.index
X_test_seq.index = X_test.index

# 拼接数据集
train_temp = pd.concat([X_train[category_cols], X_train_seq], axis=1)
test_temp = pd.concat([X_test[category_cols], X_test_seq], axis=1)

features_train_new, features_test_new, colNames_train_new, colNames_test_new = Cross_Combination(list(train_temp.columns), 
                                                                                                 train_temp, 
                                                                                                 test_temp)


# 修改index
features_train_new.index = X_train.index

# 拼接衍生特征与标签
df_temp = pd.concat([features_train_new, y_train], axis=1)

# 查看拼接后的df
df_temp.head()

new_col = list(np.abs(df_corr).sort_values(ascending = False)[1: 4].index)

train_new_temp = features_train_new[new_col]
test_new_temp = features_test_new[new_col]

features_test(train_new_temp, 
              test_new_temp, 
              X_train, 
              X_test, 
              y_train, 
              y_test, 
              category_cols, 
              numeric_cols)




# 分组统计特征
# 需要简单对此前的特征衍生过程与结果进行汇总，从而帮助我们更好的理解接下来分组统计特征衍生流程中的要点。
# 重要特征与普通特征
# 关于特征重要性，此处暂时以相关系数大小作为衡量指标，后续还将介绍其他衡量方法。

# 特征衍生现象：“强者恒强”
# 在大部分情况下，重要特征彼此的两两交叉组合、或者重要特征和普通特征的两两交叉组合而成的特征，
# 也都能够和标签展现出良好的相关性，也就是说重要特征衍生出来的特征，大概率还是重要特征；
# 而反观普通特征，只有极少一部分普通特征在和重要特征进行交叉组合后的特征才能表现出较好的特性

# 特征衍生本质：增强特征表现
# 特征衍生的根本作用是（一定程度、一定概率下）增强特征表现。

# 分组统计特征衍生的“方向性”
#  不过就交叉组合特征衍生来说（当然也包括多项式特征衍生），这种特征增强是不分方向的，次级特征的重要性（相关系数）都是继承自父特征
# 我们很难说'OnlineSecurity'和'Contract'交叉组合后的特征是强化了哪个原始特征
# “强者恒强”在交叉组合特征衍生过程中，对我们的指导意义在于需要尽可能带入重要特征进行大规模的交叉组合
# 但是并非所有的特征衍生方法都是如此，例如分组统计特征衍生。

# 而在此情况下，会不会仍然存在“强者恒强”的现象呢？如果存在，是KeyCol选取重要特征衍生的结果更好，还是分组变量选取KeyCol时衍生特征效果更好呢？

# 分组统计特征衍生其实就是围绕KeyCol进行的特征增强，也就是说，如果KeyCol本身是重要特征，则经过分组统计汇总后也能衍生出很多重要特征。
# 可以说分组统计特征衍生在所有特征衍生方法中，效率是最高的。



# 时序衍生字段分组统计
# 接下来我们进一步测试以时序字段作为KeyCol测试特征衍生的效果。首先我们将带时序衍生字段与原始数据集进行拼接：

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
ord_enc.fit(X_train[category_cols])
ord_enc.transform(X_train[category_cols])
X_train_OE = pd.DataFrame(ord_enc.transform(X_train[category_cols]), columns=category_cols)
X_train_OE.index = X_train.index
X_train_OE = pd.concat([X_train_OE, X_train[numeric_cols]], axis=1)

X_test_OE = pd.DataFrame(ord_enc.transform(X_test[category_cols]), columns=category_cols)
X_test_OE.index = X_test.index
X_test_OE = pd.concat([X_test_OE, X_test[numeric_cols]], axis=1)

 #调整index
X_train_seq.index = X_train.index
X_test_seq.index = X_test.index

# 拼接数据集
train_temp = pd.concat([X_train_OE, X_train_seq], axis=1)
test_temp = pd.concat([X_test_OE, X_test_seq], axis=1)

# 包括时序衍生变量在内的所有离散变量名
cat_temp = (category_cols + list(X_train_seq.columns)).copy()
cat_temp = list(X_train_seq.columns).copy()

# 创建容器
col_temp = cat_temp.copy()
colNames_train_new = []
colNames_test_new = []
features_train_new = []
features_test_new = []

for i in range(len(col_temp)):
    keyCol = col_temp.pop(i)
    features_train1, features_test1, colNames_train, colNames_test = Group_Statistics(keyCol,
                                                                                      train_temp,
                                                                                      test_temp,
                                                                                      col_num=numeric_cols,
                                                                                      col_cat=col_temp+category_cols, 
                                                                                      extension=True)
    
    colNames_train_new.extend(colNames_train)
    colNames_test_new.extend(colNames_test)
    features_train_new.append(features_train1)
    features_test_new.append(features_test1)
    
    col_temp = cat_temp.copy()

features_train_new = pd.concat(features_train_new, axis=1)
features_test_new = pd.concat(features_test_new, axis=1)


df_corr = pd.Series(dtype=np.float64)

for col in df_temp:
    corr = np.corrcoef(df_temp[col], df_temp['Churn'])[0, 1]
    s = pd.Series(corr, index=[col])
    df_corr = df_corr.append(s)

# 取相关系数绝对值最大的20个特征进行观察
np.abs(df_corr).sort_values(ascending = False)[: 20]

# 能够发现，原本就重要的时序衍生特征，经过分组统计衍生后仍然创造了很多重要特征，并且衍生特征和原始特征相关系数较为接近：

