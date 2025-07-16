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



# 读取数据
tcc = pd.read_csv(r'D:\pythonPractice\practice_example\TelcoCustomerChurn\WA_Fn-UseC_-Telco-Customer-Churn.csv')

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

def features_test(new_features,
                  features = features, 
                  labels = labels, 
                  category_cols = category_cols, 
                  numeric_cols = numeric_cols):
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






# Part 3.2 批量自动化特征衍生
# 特征衍生的“信息重排”的过程就是简单的围绕单个列进行变换或者围绕多个列进行组合
# new_customer字段的创建，其本质上就是围绕tenure字段进行的变换，即把所有tenure取值为1的用户都标记为1，其他用户标记为0

# 批量特征衍生并不会像手动特征衍生一样，先从思路出发、再分析数据集当前的业务背景或数据分布规律、最后再进行特征衍生，
# 而是优先考虑从方法出发，直接考虑单个列、不同的列之间有哪些可以衍生出新特征的方法，然后尽可能去衍生出更多的特征。

# 若要进行自动批量特征衍生，往往是一定需要搭配特征筛选方法的，也就是需要借助一些策略，来对批量创建的海量特征进行筛选，“去粗取精”，选出最能提升模型效果的特征

# 此外，批量特征衍生还将造成另外的一个问题，那就是很多衍生出来的特征并不具备业务层面的可解释性
# 批量特征衍生+特征筛选的策略，完全是一个“依据建模结果说话”的策略。


# 里我们先考虑如何把特征“做多”，然后再考虑如何把特征“做精”。

# 由于特征衍生应用场景复杂多变，需要综合数据体量、数据规律、现有算力等因素进行考虑，截至目前也并没有统一的第三方库能够提供完整特征衍生的方法实现



# 1.单变量特征衍生方法

# 数据重编码特征衍生
# 所有数据重编码的过程，新创建的列都可以作为一个额外的独立特征，
# 即我们在实际建模过程中，不一定是用重编码后新的列替换掉原始列，而是考虑同时保留新的特征和旧的特征，带入到下一个环节、即特征筛选
# 如果数据重编码后的特征是有效的，则自然会被保留，否则则会被剔除。

# 这么一来，我们或许就不用考虑在当前模型下是否需要进行数据重编码，而是无论是否需要，都先进行重编码、并同时保留原特征和衍生出来的新的特征

# 连续变量数据重编码方法
# 标准化:0-1标准化/Z-Score标准化
# 离散化：等距分箱/等频分箱/聚类分箱

# 离散变量数据重编码方法
# 自然数编码/字典编码
# 独热编码/哑变量变换

# 在同时保留原始列和重编码的列时，极有可能出现原始列和重编码的列都是有效特征的情况

# 2.高阶多项式特征衍生
# 对于单独的变量来说，除了可以通过重编码进行特征衍生外，还可以通过多项式进行特征衍生，
# 即创建一些自身数据的二次方、三次方数据等。

# 用sklearn中的PolynomialFeatures评估器来实现该过程。
# 该评估器不仅能够非常便捷的实现单变量的多项式衍生，
# 也能够快速实现多变量组合多项式衍生，且能够与机器学习流集成，也便于后续的超参数搜索方法的使用。

from sklearn.preprocessing import PolynomialFeatures
x1 = np.array([1, 2, 4, 1, 3])
PolynomialFeatures(degree=5).fit_transform(x1.reshape(-1, 1))
# 一般来说单特征的多项式往往是针对连续变量会更有效，但再在些情况下也可以将其用于离散型随机变量。


# 3.特征衍生准则
# 往往我们需要有些判断，即哪些情况下朝什么方向进行特征衍生是最有效的

# 特征衍生选择依据
# 此处我们假设实际构建的模型以集成学习为主：
# 先考虑分类变量的独热编码，并同时保留原始变量与独热编码衍生后的变量。
# 同时也需要注意的是有两种情况不适用于使用独热编码
# 其一是分类变量取值水平较多（例如超过10个取值），此时独热编码会造成特征矩阵过于稀疏，从而影响最终建模效果
# 二则是如果该离散变量参与后续多变量的交叉衍生（稍后会介绍），则一般需再对单独单个变量进行独热编码；

# 优先考虑连续变量的数据归一化，尽管归一化不会改变数据集分布，即无法通过形式上的变换增加树生长的多样性，
# 但归一化能够加快梯度下降的执行速度，加快迭代收敛的过程；

# 在连续变量较多的情况下，可以考虑对连续变量进行分箱
# 具体分箱方法优先考虑聚类分箱，若数据量过大，可以使用MiniBatch K-Means提高效率，或者也可以简化为等频率/等宽分箱；

# 不建议对单变量使用多项式衍生方法，相比单变量的多项式衍生，带有交叉项的多变量的多项式衍生往往效果会更好。


# 二、双变量特征衍生方法
# 两个特征组合成新的字段我们会称其为双变量（或者双特征）交叉衍生，而如果涉及到多个字段组合，则会称其为多变量交叉衍生

# 一般来说，双变量特征衍生是目前常见特征衍生方法中最常见、同样也是效果最好的一类方法，这也是我们接下来要重点介绍的方法。
# 而多变量特征衍生，除了四则运算（尤其以加法居多）的组合方法外，
# 其他衍生方法随着组合的字段增加往往会伴随非常严重的信息衰减，
# 因此该类方法除特定场合外一般不会优先考虑使用


# 1.四则运算特征衍生
# 该过程非常简单，就是单纯的选取两列进行四则运算
# 一般来说，四则运算特征衍生的使用场景较为固定，主要有以下三个：
# 在某些数据集中，我们需要通过四则运算来创建具有明显业务含义的补充字段，
# 例如在上述电信用户流失数据集中，我们可以通过将总消费金额除以用户入网时间，即可算出用户平均每月的消费金额
# 其二,往往在特征衍生的所有工作结束后，我们会就这一系列衍生出来的新字段进行四则运算特征衍生，作为数据信息的一种补充
# 其三，在某些极为特殊的字段创建过程中使用，例如竞赛中常用的黄金组合特征、流量平滑特征（稍后会重点讨论）等，需要使用四则运算进行特征衍生。


# 2.交叉组合特征衍生
# 指的是不同分类变量不同取值水平之间进行交叉组合，从而创建新字段的过程。
# 例如此前我们创建的老年且经济不独立的标识字段，实际上就是是否是老年人字段（SeniorCitizen）和是否经济独立字段（Dependents）两个字段交叉组合衍生过程中的一个：

# 交叉组合后衍生的特征个数是参数交叉组合的特征的取值水平之积，因此交叉组合特征衍生一般只适用于取值水平较少的分类变量之间进行
# 若是分类变量或者取值水平较多的离散变量彼此之间进行交叉组合，则会导致衍生特征矩阵过于稀疏

# 手动实现
# 仍然以telco数据集为例，尝试围绕'SeniorCitizen'、'Partner'、'Dependents'字段进行两两交叉组合衍生

# 提取目标字段
colNames = ['SeniorCitizen', 'Partner', 'Dependents']

# 单独提取目标字段的数据集
features_temp = features[colNames]
features_temp.head(5)

# 创建空列表用于存储衍生后的特征名称和特征
colNames_new_l = []
features_new_l = []

# 创建衍生特征列名称及特征本身
for col_index, col_name in enumerate(colNames):
    for col_sub_index in range(col_index+1, len(colNames)):
        
        newNames = col_name + '&' + colNames[col_sub_index]
        colNames_new_l.append(newNames)
        
        newDF = pd.Series(features[col_name].astype('str') 
                          + '&'
                          + features[colNames[col_sub_index]].astype('str'), 
                          name=col_name)
        features_new_l.append(newDF)


features_new = pd.concat(features_new_l, axis=1)
features_new.columns = colNames_new_l

# 我们创建了3个4分类的变量，我们可以直接将其带入进行建模，
# 但需要知道的是这些四分类变量并不是有序变量，因此往往我们需要进一步将这些衍生的变量进行独热编码，然后再带入模型：

enc = preprocessing.OneHotEncoder()

# 借助此前定义的列名称提取器进行列名称提取

# 最后创建一个完整的衍生后的特征矩阵
# 最后创建一个完整的衍生后的特征矩阵
features_new_af = pd.DataFrame(enc.fit_transform(features_new).toarray(), 
                               columns = cate_colName(enc, colNames_new_l, drop=None))

print(features_new_af.head(5))

# 函数封装
def Binary_Cross_Combination(colNames, features, OneHot=True):
    """
    分类变量两两组合交叉衍生函数
    
    :param colNames: 参与交叉衍生的列名称
    :param features: 原始数据集
    :param OneHot: 是否进行独热编码
    
    :return：交叉衍生后的新特征和新列名称
    """
    
    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []
    
    # 提取需要进行交叉组合的特征
    features = features[colNames]
    
    # 逐个创造新特征名称、新特征
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            
            newNames = col_name + '&' + colNames[col_sub_index]
            colNames_new_l.append(newNames)
            
            newDF = pd.Series(features[col_name].astype('str')  
                              + '&'
                              + features[colNames[col_sub_index]].astype('str'), 
                              name=col_name)
            features_new_l.append(newDF)
    
    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l
    
    # 对新特征矩阵进行独热编码
    if OneHot == True:
        enc = preprocessing.OneHotEncoder()
        enc.fit_transform(features_new)
        colNames_new = cate_colName(enc, colNames_new_l, drop=None)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=colNames_new)
        
    return features_new, colNames_new

#本节定义的特征衍生函数都将创建衍生列的特征名称，同时输出的数据也是衍生后的新的特征矩阵，而非和原数据拼接后的结果

features_new, colNames_new = Binary_Cross_Combination(colNames, features)
print(features_new.head(5))

# 当然，完成衍生特征矩阵创建后，还需要和原始数据集进行拼接，此处拼接过程较为简单，直接使用concat函数即可：
df_temp = pd.concat([features, features_new], axis=1)


# 这里我们不着急带入新的特征进入模型进行效果测试，对于大多数批量创建特征的方法来说，创建的海量特征往往无效特征占绝大多数

#  在实际使用过程中，双变量的交叉衍生是最常见的特征衍生方法，也是第一梯队优先考虑的特征衍生的策略


# 3.分组统计特征衍生
# 一种同样非常常用的特征衍生方法：分组统计特征衍生方法
# A特征根据B特征的不同取值进行分组统计，统计量可以是均值、方差等针对连续变量的统计指标，
# 也可以是众数、分位数等针对离散变量的统计指标，
# 例如我们可以计算不同入网时间用户的平均月消费金额、消费金额最大值、消费金额最小值等

# 在实际执行分组统计特征衍生的过程中（假设是A特征根据B特征的不同取值进行分组统计），有以下几点需要注意：

# 首先，一般来说A特征可以是离散变量也可以是连续变量，
# 而B特征必须是离散变量，且最好是一些取值较多的离散变量（或者固定取值的连续变量）
# 例如本数据集中的tenure字段，总共有73个取值。主要原因是如果B特征取值较少，则在衍生的特征矩阵中会出现大量的重复的行；

# 其次，在实际计算A的分组统计量时，可以不局限于连续特征只用连续变量的统计量、离散特征只用离散的统计量，完全可以交叉使用
# A是离散变量，我们也可以分组统计其均值、方差、偏度、峰度等，连续变量也可以统计众数、分位数等。很多时候，更多的信息组合有可能会带来更多的可能性；

# 其三，有的时候分组统计还可以用于多表连接的场景，例如假设现在给出的数据集不是每个用户的汇总统计结果，而是每个用户在过去的一段时间内的行为记录，
# 则我们可以根据用户ID对其进行分组统计汇总

# 其四，很多时候我们还会考虑进一步围绕特征A和分组统计结果进行再一次的四则运算特征衍生，
# 例如用月度消费金额减去分组均值，则可以比较每一位用户与相同时间入网用户的消费平均水平的差异，
# 围绕衍生特征再次进行衍生，我们将其称为统计演变特征，也是分组汇总衍生特征的重要应用场景：

# 手动实现
# 这里我们可以优先考虑借助Pandas中的groupby方法来实现，首先简单回归groupby方法的基本使用，这里我们提取'tenure'、'SeniorCitizen'、'MonthlyCharges'三列来尝试进行单列聚合和多列聚合：

# 提取目标
colNames = ['tenure', 'SeniorCitizen', 'MonthlyCharges']
# 单独提取目标字段的数据集
features_temp = features[colNames]
# 在不同tenure取值下计算其他变量分组均值的结果
features_temp.groupby('tenure').mean() # 根据tenure的取值进行分组

# 在不同tenure取值下计算其他变量分组标准差的结果
features_temp.groupby('tenure').std()

# 在'tenure'、'SeniorCitizen'交叉取值分组下，计算组内月度消费金额均值
features_temp.groupby(['tenure', 'SeniorCitizen']).mean()

# 当然，groupby也支持同时输入多个统计量进行汇总计算，此时推荐使用agg方法来进行相关操作:

# 分组汇总字段
colNames_sub =  ['Senior', 'Monthly']

# 创建空字典
aggs = {}

# 字段汇总统计量设置
for col in colNames_sub:
    aggs[col] = ['mean', 'min', 'max']

# 创建新的列名
cols = ['tenure']

for key in aggs.keys():
    cols.extend(key+'_'+stat for stat in aggs[key])

# 接下来我们创建新特征：
features_new = features_temp.groupby('tenure').agg(aggs).reset_index()
print(features_new.head(5))

# 重新设置列名称
features_new.columns = cols
print(features_new.head(5))

df_temp = pd.merge(features, features_new, how='left',on='tenure') 
df_temp.head()


# 常用统计量补充
# 可用于连续性变量的统计量如下：
#  mean/var：均值、方差；
#  max/min：最大值、最小值；
#  skew：数据分布偏度，小于零时左偏，大于零时右偏；

a = np.array([1, 2, 3, 2, 5, 1], [0, 0, 0, 1, 1, 1])
df = pd.DataFrame(a.T, columns=['x1', 'x2'])

aggs = {'x1': ['mean', 'var', 'max', 'min', 'skew']}
df.groupby('x2').agg(aggs).reset_index()



# 常用的分类变量的统计量如下，当然除了偏度外，其他连续变量的统计量也是可用于分类变量的：
# - median：中位数；
# - count：个数统计；
# - nunique：类别数；
# - quantile：分位数

df = pd.DataFrame({'x1':[1, 3, 4, 2, 1], 'x2':[0, 0, 1, 1, 1]})
aggs = {'x1': ['median', 'count', 'nunique']}
df.groupby('x2').agg(aggs).reset_index()

def Binary_Group_Statistics(keyCol, 
                            features, 
                            col_num=None, 
                            col_cat=None, 
                            num_stat=['mean', 'var', 'max', 'min', 'skew', 'median'], 
                            cat_stat=['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'], 
                            quant=True):
    """
    双变量分组统计特征衍生函数
    
    :param keyCol: 分组参考的关键变量
    :param features: 原始数据集
    :param col_num: 参与衍生的连续型变量
    :param col_cat: 参与衍生的离散型变量
    :param num_stat: 连续变量分组统计量
    :param cat_num: 离散变量分组统计量  
    :param quant: 是否计算分位数  

    :return：交叉衍生后的新特征和新特征的名称
    """
    
    # 当输入的特征有连续型特征时
    if col_num != None:
        aggs_num = {}
        colNames = col_num
        
        # 创建agg方法所需字典
        for col in col_num:
            aggs_num[col] = num_stat 
            
        # 创建衍生特征名称列表
        cols_num = [keyCol]
        for key in aggs_num.keys():
            cols_num.extend([key+'_'+keyCol+'_'+stat for stat in aggs_num[key]])
            
        # 创建衍生特征df
        features_num_new = features[col_num+[keyCol]].groupby(keyCol).agg(aggs_num).reset_index()
        features_num_new.columns = cols_num 
        
        # 当输入的特征有连续型也有离散型特征时
        if col_cat != None:        
            aggs_cat = {}
            colNames = col_num + col_cat

            # 创建agg方法所需字典
            for col in col_cat:
                aggs_cat[col] = cat_stat

            # 创建衍生特征名称列表
            cols_cat = [keyCol]
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+keyCol+'_'+stat for stat in aggs_cat[key]])    

            # 创建衍生特征df
            features_cat_new = features[col_cat+[keyCol]].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat
    
            # 合并连续变量衍生结果与离散变量衍生结果
            df_temp = pd.merge(features_num_new, features_cat_new, how='left',on=keyCol)
            features_new = pd.merge(features[keyCol], df_temp, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_num + cols_cat
            colNames_new.remove(keyCol)
            colNames_new.remove(keyCol)
         
        # 当只有连续变量时
        else:
            # merge连续变量衍生结果与原始数据，然后删除重复列
            features_new = pd.merge(features[keyCol], features_num_new, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_num
            colNames_new.remove(keyCol)
    
    # 当没有输入连续变量时
    else:
        # 但存在分类变量时，即只有分类变量时
        if col_cat != None:
            aggs_cat = {}
            colNames = col_cat

            for col in col_cat:
                aggs_cat[col] = cat_stat

            cols_cat = [keyCol]
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+keyCol+'_'+stat for stat in aggs_cat[key]])    

            features_cat_new = features[col_cat+[keyCol]].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat            
             
            features_new = pd.merge(features[keyCol], features_cat_new, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_cat
            colNames_new.remove(keyCol) 
    
    if quant:
        # 定义四分位计算函数
        def q1(x):
            """
            下四分位数
            """
            return x.quantile(0.25)

        def q2(x):
            """
            上四分位数
            """
            return x.quantile(0.75)

        aggs = {}
        for col in colNames:
            aggs[col] = ['q1', 'q2']

        cols = [keyCol]
        for key in aggs.keys():
            cols.extend([key+'_'+keyCol+'_'+stat for stat in aggs[key]])    

        aggs = {}
        for col in colNames:
            aggs[col] = [q1, q2]    

        features_temp = features[colNames+[keyCol]].groupby(keyCol).agg(aggs).reset_index()
        features_temp.columns = cols

        features_new = pd.merge(features_new, features_temp, how='left',on=keyCol)
        features_new.loc[:, ~features_new.columns.duplicated()]
        colNames_new = colNames_new + cols
        colNames_new.remove(keyCol)     
    
    features_new.drop([keyCol], axis=1, inplace=True)
        
    return features_new, colNames_new


# 一般来说在进行分组统计时，需要注意某些统计指标在计算过程中可能造成缺失值，需要在执行完特征衍生后再进行缺失值查找：

df.isnull().sum()




# 4.多项式特征衍生
# 双变量的多项式衍生会比单变量多项式衍生更有效果，该过程并不复杂，只是在单变量多项式衍生基础上增加了交叉项的计算
# 般来说双变量多项式衍生只适用于两个连续变量之间，一个连续变量一个离散变量或者两个离散变量进行多项式衍生意义不大
# 在选取特征进行多项式衍生的过程中，往往我们不会随意组合连续变量来进行多项式衍生，而是只针对我们判断非常重要的特征来进行多项式衍生。
# 就这点而言，多项式衍生和四则运算衍生非常类似，其使用场景背后的基本思路也完全一致：强化重要特征的表现形式；
# 往往我们只会衍生3阶左右，极少数情况会衍生5-10阶。
# 而伴随着多项式阶数的增加，也需要配合一些手段来消除数值绝对值爆炸或者衰减所造成的影响，
# 例如对数据进行归一化处理等；


# 实现过程
from sklearn.preprocessing import PolynomialFeatures
df = pd.DataFrame({'X1':[1, 2, 3], 'X2':[2, 3, 4]})

PolynomialFeatures(degree=2, include_bias=False).fit_transform(df)
# 重点需要关注以下两个参数
# interaction_only：默认False，如果选择为True，则表示只创建交叉项；
# include_bias：默认为True，即考虑计算特征的0次方，除了需要人工捕捉截距，否则建议修改为False。全是1的列不包含任何有效信息；

def Binary_PolynomialFeatures(colNames, degree, features):
    """
    连续变量两变量多项式衍生函数
    
    :param colNames: 参与交叉衍生的列名称
    :param degree: 多项式最高阶
    :param features: 原始数据集
    
    :return：交叉衍生后的新特征和新列名称
    """
    
    
    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []
    
    # 提取需要进行多项式衍生的特征
    features = features[colNames]
    
    # 逐个进行多项式特征组合
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            col_temp = [col_name] + [colNames[col_sub_index]]
            array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features[col_temp])
            features_new_l.append(pd.DataFrame(array_new_temp[:, 2:]))
    
            # 逐个创建衍生多项式特征的名称
            for deg in range(2, degree+1):
                for i in range(deg+1):
                    col_name_temp = col_temp[0] + '**' + str(deg-i) + '*'+ col_temp[1] + '**' + str(i)
                    colNames_new_l.append(col_name_temp)
            
    
    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l
    
    return features_new, colNames_new




# 关键特征衍生策略
# 于某些“特殊”的特征，是无法通过上述自动化特征衍生方法进行更深入的有效信息挖掘的，
# 在机器学习领域中，这些“特殊”的特征就是时序特征和文本特征，例如此前数据集中的tenure字段就属于时序字段

# 对于时序特征，往往都包含更多更复杂的信息，例如tenure字段，
# 截至目前我们只知道tenure字段代表用户入网时间，并且数值越大表示用户入网时间越早，并且tenure字段和标签呈现负相关，
# 即入网时间越早的用户（tenure字段取值越大）越不容易流失：


# 时序特征衍生方法
# tenure时序字段分析
# 对于时序字段来说，也绝不仅仅是一串记录了事件发生时间的数字或者字符串这么简单，
# 其背后往往可能隐藏着非常多极有价值的信息，
# 例如在很多场景下数据集的标签都会具有季节性波动规律，
# 对应到上述数据集就是用户流失很有可能与季节、月份相关。而要围绕这点进行分析，
# 我们首先就需要将tenure数值型字段转化为年-月-季度的基本格式。


# 只需要借助numpy的广播特性，围绕tenure列进行除法和取余计算即可。
# 数值越大代表距离开始统计的时间越远，即tenure=0表示开始统计的第一个月、即2014年1月，
# 而tenure=72则表示结束统计的最后一个月，即2020年1月：

feature_seq = pd.DataFrame()
feature_seq['tenure'] = features['tenure']

# 年份衍生
feature_seq['tenure_year'] = ((72 - features['tenrue'])//12) + 2014

# 月份衍生
feature_seq['tenure_month'] = (72 - features['tenure']) % 12 + 1

# 季度衍生
feature_seq['tenure_quarter'] = ((feature_seq['tenure_month']-1) // 3) + 1


# 至此，我们就根据tenure字段完成了更细粒度的时间划分。接下来我们就从这些更细粒度刻画时间的字段入手进行分析。

#  根据此前的分析我们知道，tenure和用户流失呈现明显的负相关，即tenure取值越大（越是老用户），用户流失的概率就越小

# 接下来我们进一步分析用户流失是否和入网年份、月份和季度有关
feature_seq['Churn'] = labels

# 我们可以首先简单尝试计算相关系数
# 在未进行独热编码之前，相关系数的计算实际上会将拥有多个取值水平的分类变量视作连续变量来进行计算。
# 例如quarter取值为[1,4]，则相关系数的计算过程是quarter取值越大、流失概率如何变化：
# 出于更严谨的角度考虑，我们还是需要将年、月、季度等字段进行独热编码

seq_new = ['tenure_year', 'tenure_month', 'tenure_quarter']
feature_seq_new = feature_seq[seq_new]
enc = preprocessing.OneHotEncoder()
enc.fit_transform(feature_seq_new)

# 借助此前定义的列名称提取器进行列名称提取
cate_colName(enc, seq_new, drop=None)

# 创建带有列名称的独热编码之后的df
features_seq_new = pd.DataFrame(enc.fit_transform(feature_seq_new).toarray(),
                                columns = cate_colName(enc, seq_new, drop=None))

# 添加标签列
features_seq_new['Churn'] = labels

# 接下来即可通过柱状图来观察用户不同入网时间与流失率之间的相关关系：
sns.set()
plt.figure(figsize=(15, 8), dpi=200)

feature_seq_new.corr()['Churn'].sort_values(ascending = False).plot(kind='nar')

# 能够看出，2019年入网用户普遍流失率更大、而2014年入网用户普遍流失率更小
# 当然这个和tenure字段与标签的相关性表现出来的规律是一致的，即越是新入网的用户、流失概率越大

# 尽管该规律真实有效，但在实际使用过程中还是要注意区分场景，
# 如果是针对既有用户（例如过去6年积累下来的所有用户）进行流失率预测，则入网时长就将是重要特征，
# 但如果是要针对每个月入网的新用户进行流失率的实时预测，则入网年份可能就不是那么重要了

# 我们都可以将这些相关性较强的字段带入模型进行建模
abs(feature_seq_new.corr()['Churn']).sort_values(ascending = False)

features_new_cols1 = list(abs(features_seq_new.corr()['Churn']).sort_values(ascending = False)[1: 3].index)

features_new1 = features_seq_new[features_new_cols1]


# 这里我们尝试使用此前定义的features_test函数，用逻辑回归模型对比测试带入新特征之后的模型效果：
features_test(features_new1,
              features = features, 
              labels = labels, 
              category_cols = category_cols, 
              numeric_cols = numeric_cols)

# 能够看出，模型效果有了非常显著的提升（原数据集情况下交叉验证的平均准确率为0.8042）。
# 当然，这里我们进一步来进行分析，其实这两个年份字段的根本作用就是给了新入网和最早入网的用户一个标识，
# 本质上也是强化了越早入网越不容易流失这一规律的外在表现



# 时序字段的二阶特征衍生
# 在上述例子中，我们已经发现在时序字段衍生字段中，季度是相对重要的特征，
# 那么接下来我们接下来我们可以进一步将季度字段与其他原始字段进行交叉组合、分组统计汇总等，
# 去进行进一步的特征衍生。

# 在其他很多场景下时序字段的特征衍生与分析可能会非常复杂

# Pandas中的时间记录格式
# 本案例中tenure字段是经过自然数编码后的字段，时间是以整数形式进行呈现，
# 而对于其他很多数据集，时序字段往往记录的就是时间是真实时间，
# 并且是精确到年-月-日、甚至是小时-分钟-秒的字段，例如"2022-07-01;14:22:01"

t = pd.DataFrame()
t['time'] = ['2022-01-03;02:31:52',
             '2022-07-01;14:22:01', 
             '2022-08-22;08:02:31', 
             '2022-04-30;11:41:31', 
             '2022-05-02;22:01:27']

# 对于这类object对象，我们可以字符串分割的方式对其进行处理，
# 此外还有一种更简单通用同时也更有效的方法：将其转化为datetime64格式,这也是pandas中专门用于记录时间对象的格式

# datetime64来说有两种子类型，分别是datetime64[ns]毫秒格式与datetime64[D]日期格式。

# 无论是哪种格式，我们可以通过pd.to_datetime函数对其进行转化
t['time'] = pd.datatime(t['time'])
# 该函数会自动将目标对象转化为datetime64[ns]类型
# 当然，如果我们的时间记录格式本身只精确到日期、并没有时分秒，
# 通过pd.to_datetime函数仍然会转化为datetime64[ns]类型，只是此时显示的时间中就没有时分秒：


# 就类似于浮点数与整数，更高的精度往往会导致更大的计算量，对于本身只是精确到日期的时间记录格式，
# 我们其实可以用另一种只能精确到天的时间数据格式进行记录，也就是datetime64[D]类型。

# 在pd.to_datetime函数使用过程中是无法直接创建datetime64[D]类型对象的，
# 我们需要使用.value.astype('datetime64[D]')的方法对其进行转化，
# 但是需要注意，这个过程最终创建的对象类型是array，而不再是Series了：
t['time'].values.astype('datatime64[D]')

# 能够看出，array是支持datetime64[ns]和datetime64[D]等多种类型存储的，
# 但对于pandas来说，只支持以datetime64[ns]类型进行存储，
# 哪怕输入的对象类型是datetime64[D]，在转化为Series时仍然会被转化为datetime64[ns]：


# 在绝大多数情况下，我们都建议采用pandas中datetime类型记录时间（尽管从表面上来看也可以用字符串来表示时间），
# 这也将极大程度方便我们后续对关键时间信息的提取。


# 时序字段的通用信息提取方式
# 可以通过一些dt.func方式来提取时间中的关键信息，如年、月、日、小时、季节、一年的第几周等，常用方法如下所示：
# dt.year	提取年
# dt.month	提取月
t['time'].dt.second

#接下来我们用不同的列记录这些具体时间信息：
t['year']        =  t['time'].dt.year
t['month']       =  t['time'].dt.month
t['day']         =  t['time'].dt.day
t['hour']        =  t['time'].dt.hour
t['minute']      =  t['time'].dt.minute
t['second']      =  t['time'].dt.second


# 时序字段的其他自然周期提取方式
# 对于时序字段，往往我们会尽可能的对其进行自然周期的划分，
# 然后在后续进行特征筛选时再对这些衍生字段进行筛选
# 除了季度，诸如全年的第几周、一周的第几天，甚至是日期是否在周末，具体事件的时间是在上午、下午还是在晚上等，都会对预测造成影响

# 对于这些自然周期提取方法，有些自然周期可以通过dt的方法自动计算，另外则需要手动进行计算。
# dt.quarter	提取季度
# dt.weekofyear	提取年当中的周数
# dt.dayofweek, dt.weekday	提取周几

# 这里需要注意，每周一是从0开始计数，这里我们可以手动+1，对其进行数值上的修改：
t['time'].dt.dayofweek + 1

t['quarter']     =  t['time'].dt.quarter
t['weekofyear']  =  t['time'].dt.weekofyear
t['dayofweek']   =  t['time'].dt.dayofweek + 1

# 接下来继续创建是否是周末的标识字段：
t['weekend'] = (t['dayofweek'] > 5).astype(int)

# 进一步创建小时所属每一天的周期，凌晨、上午、下午、晚上，周期以6小时为划分依据：
t['hour_section'] = (t['hour'] // 6).astype(int) 
# 至此，我们就完成了围绕时序字段的详细信息衍生（年月日、时分秒单独提取一列），
# 以及基于自然周期划分的时序特征衍生（第几周、周几、是否是周末、一天中的时间段）。


# 时序字段特征衍生的本质与核心思路
# 对用户进行分组之所以能够帮助模型进行建模与训练，
# 其根本原因也是因为有的时候，同一组内（或者是多组交叉）的用户会表现出相类似的特性（或者规律），
# 从而能够让模型更快速的对标签进行更准确的预测


# 时序字段衍生的核心思路：自然周期和业务周期
#  进行了细节时间特征的衍生之后（划分了年月日、时分秒之后），
# 接下来的时序特征衍生就需要同时结合自然周期和业务周期两个方面进行考虑。

# 所谓自然周期，指的是对于时间大家普遍遵照或者约定俗成的一些规定，例如工作日周末、一周七天、一年四个季度等
# 其实还可以根据一些业务周期来进行时序特征的划分，
# 例如对于部分旅游景点来说，暑假是旅游旺季，
# 并且很多是以家庭为单位进行出游（学生暑假），
# 因此可以考虑单独将8、9月进行标记，期间记录的用户会有许多共性
# 6月、11月是打折季

# 但是，在另外一些场景下，例如某线下超市的周五，可能就是一个需要重点关注的时间，
# 不仅是因为临近周末很多客户会在下班后进行集中采购、而且很多超市有“黑五”打折的习惯，
# 如果是进行超市销售额预测，是否是周五可能就需要单独标注出来，形成独立的一列（该列被包含在dayofweek的衍生列中）。

#  时序字段的补充衍生方法：关键时间点的时间差值衍生
# 实际操作过程中需要首先人工设置关键时间点，然后计算每条记录和关键时间点之间的时间差，具体时间差的衡量以天和月为主，当然也可以根据数据集整体的时间跨度为前提进行考虑
# 关键时间点一般来说可以是数据集记录的起始时间、结束时间、距今时间，也可以是根据是根据业务或者数据集本身的数据规律，推导出来的关键时间点。

# 在pandas中我们也可以非常快捷的进行时间差值的计算，我们可以直接将datetime64类型的两列进行相减：

# 果是用一列时间减去某个时间的话，我们需要先将这个单独的时间转化为时间戳，然后再进行相减。
# 所谓时间戳，可以简单理解为单个时间的记录格式，在pandas中可以借助pd.Timestamp完成时间戳的创建：

p1 = '2022-01-03;02:31:52'
t['time'] - pd.Timestamp(p1)

# 在一个datetime64[ns]类型的Series中，每个元素都是一个时间戳：
# 对于Timestamp对象类型，我们同样也可以用year、month、second等时间信息：
pd.Timestamp(p1).year, pd.Timestamp(p1).second

# 接下来，我们重点讨论时间相减之后的结果。能够发现，上述时间差分的计算结果中，最后返回结果的对象类型是timedelta64[ns]，
# 其中timedelta表示时间差值计算结果

# 可以进一步调用timedelta64[ns]的一些属性，来单独提取相差的天数和相差的秒数：
td = t['time'] - pd.Timestamp(p1)
t['time_diff'] = td
td.dt.days

# 据此我们可以借助相差的天数进一步计算相差的月数
np.round(td.dt.days / 30).astype('int')

t['time_diff_days'] = td.dt.days
t['time_diff_seconds'] = td.dt.seconds

# 可能会希望计算真实时间差的秒数以及小时数，此时应该怎么做呢？
# timedelta64同样也有timedelta64[h]、timedelta64[s]等对象类型，
# 但pandas中只支持timedelta64[ns]对象类型，
# 我们仍然可以通过.values.astype的方法将时间运算差值转化为timedelta64[h]或timedelta64[s]的array
# 但再次转化成Series时又会变为timedelta64[ns]：
t['time_diff_h'] = td.values.astype('timedelta64[h]').astype('int')
t['time_diff_s'] = td.values.astype('timedelta64[s]').astype('int')

# 当然，这些已经转化为整数类型的对象是无法（当然也没有必要）进行任何时序方面的操作、
# 如进行自然周期划分、进行时间差值计算等。

# 而对于关键时间点的时间戳的提取，如数据集起止时间的计算，可以通过min/max方法实现：
t['time'].max()
t['time'].min()

# 同时，我们可以通过如下方式获取当前时间
import datetime
datetime.datetime.now()

# 在默认情况下，获得的是精确到毫秒的结果：
pd.Timestamp(datetime.datetime.now())

# 当然，我们也可以通过如下方式自定义时间输出格式，并借此输出指定精度的时间：
print(datetime.datetime.now().strftime('%Y-%m-%d'))
print(pd.Timestamp(datetime.datetime.now().strftime('%Y-%m-%d')))

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(pd.Timestamp(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# 有了自动获取这些关键时间戳的方法，我们就可以进一步带入进行时间差值的计算了。至此，我们就完成了所有时间特征衍生的简单实践。

# 1.4 时序字段特征衍生方法汇总
def timeSeriesCreation(timeSeries, timeStamp=None, precision_high=False):
    """
    时序字段的特征衍生
    
    :param timeSeries：时序特征，需要是一个Series
    :param timeStamp：手动输入的关键时间节点的时间戳，需要组成字典形式，字典的key、value分别是时间戳的名字与字符串
    :param precision_high：是否精确到时、分、秒
    :return features_new, colNames_new：返回创建的新特征矩阵和特征名称
    """
    
    # 创建衍生特征df
    features_new = pd.DataFrame()
    
    # 提取时间字段及时间字段的名称
    timeSeries = pd.to_datetime(timeSeries)
    colNames = timeSeries.name
    
    # 年月日信息提取
    features_new[colNames+'_year'] = timeSeries.dt.year
    features_new[colNames+'_month'] = timeSeries.dt.month
    features_new[colNames+'_day'] = timeSeries.dt.day
    
    if precision_high != False:
        features_new[colNames+'_hour'] = timeSeries.dt.hour
        features_new[colNames+'_minute'] = timeSeries.dt.minute
        features_new[colNames+'_second'] = timeSeries.dt.second
    
    # 自然周期提取
    features_new[colNames+'_quarter'] = timeSeries.dt.quarter
    features_new[colNames+'_weekofyear'] = timeSeries.dt.weekofyear
    features_new[colNames+'_dayofweek'] = timeSeries.dt.dayofweek + 1
    features_new[colNames+'_weekend'] = (features_new[colNames+'_dayofweek'] > 5).astype(int) 
    
    if precision_high != False:
        features_new['hour_section'] = (features_new[colNames+'_hour'] // 6).astype(int) 
    
    # 关键时间点时间差计算
    # 创建关键时间戳名称的列表和时间戳列表
    timeStamp_name_l = []
    timeStamp_l = []
    
    if timeStamp != None:
        timeStamp_name_l = list(timeStamp.keys())
        timeStamp_l = [pd.Timestamp(x) for x in list(timeStamp.values())]
    
    # 准备通用关键时间点时间戳
    time_max = timeSeries.max()
    time_min = timeSeries.min()
    time_now = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    timeStamp_name_l.extend(['time_max', 'time_min', 'time_now'])
    timeStamp_l.extend([time_max, time_min, time_now])
    
    # 时间差特征衍生
    for timeStamp, timeStampName in zip(timeStamp_l, timeStamp_name_l):
        time_diff = timeSeries - timeStamp
        features_new['time_diff_days'+'_'+timeStampName] = time_diff.dt.days
        features_new['time_diff_months'+'_'+timeStampName] = np.round(features_new['time_diff_days'+'_'+timeStampName] / 30).astype('int')
        if precision_high != False:
            features_new['time_diff_seconds'+'_'+timeStampName] = time_diff.dt.seconds
            features_new['time_diff_h'+'_'+timeStampName] = time_diff.values.astype('timedelta64[h]').astype('int')
            features_new['time_diff_s'+'_'+timeStampName] = time_diff.values.astype('timedelta64[s]').astype('int')
    
    colNames_new = list(features_new.columns)
    return features_new, colNames_new

timeStamp = {'p1':'2022-03-25 23:21:52', 'p2':'2022-02-15 08:51:02'}
features_new, colNames_new = timeSeriesCreation(timeSeries=t['time'], timeStamp=timeStamp, precision_high=True)

# 时间序列模型简介
# 时间序列是一种用于进行回归问题建模的模型，整个建模过程只有时间这一个特征，
# 也就是说时间序列的模型预测过程就是希望通过时间这一个特征去对标签进行连续型数值预测。

# 自相关，指的是标签自己和自己相关，并且指的是某个时间点的自己和之前一段时间的自己是相关的
# 。例如在上述数据集中，每个标签的取值，其实就和一个季度前的自己有极大的相关性（甚至取值都是一样的）
# 而标签一定要在时间维度上呈现出一定的自相关性，才能进一步带入时间序列模型进行建模预测

# 完全的自相关的情况还是比较少，更常见的情况是存在自相关、但也并不是完全自相关

# 哪怕间隔时间相同，随着时间变化，“过去的自己”和“现在的自己”差异也在逐渐增加，这种变化，
# 也被称为时间序列中的趋势。对应到当前数据中来，就是整体序列的变化趋势是稳步增加的，
# 即销售额尽管受到不同季度的季度周期影响，但整体是呈现上涨趋势的。

# 而时间序列中存在一些“趋势”，也是非常常见的一种情况。当然，除了一些季节性波动、长期趋势以外，
# 有些时候还会出现一些偶然因素导致的时间序列发生变化，例如数据集如下所示：

# 对于时间序列模型来说，其根本作用就是去捕捉目标变量（标签）在时间维度中所呈现出来的季节波动、
# 长期规律以及循环规律，然后再使用一些随机事件对标签取值进行修正，最后得出预测结果。
# 当然，判断某数据集是否适合进行时间序列的建模，首先我们需要对其进行自相关性（以及偏自相关性）检验。



# 时间序列模型本身的有效性
#  实际上在我们的生产生活中，是存在很大一部分预测场景、能用且只能用时间序列模型来进行预测的。
# 股票预测，患病总人次。

# 在这些预测场景中，我们很难采用一般的机器学习模型进行建模，其中最难的地方在于无法有效的提取特征。
# 对于区域患病人次预测项目来说，影响一个地区患病总人数的有效特征是非常难以提取及评估的

# 而另一方面，尽管区域患病总人次会受到诸多不确定因素影响，但在一个较长的时间段中，整体患病人次在时间维度上的分布还是有一定规律的

# 时间序列的模型局限
# 时间序列本身的局限也非常明显，或者说时间序列本身就是一种只适用于特殊情况的模型，
# 即当一系列的不确定性因素产生了某种综合性的确定性的影响之后，时间序列能够非常快速的对其进行规律挖掘，
# 但如果预测项目可以有效提取特征，那么机器学习模型肯定是潜力更大的模型。

# 如果标签并未在时间维度上呈现周期性波动、趋势性变化，也没有自相关性，那么时间序列模型也是无法构建的。
# 哪怕是数据分布规律符合时间序列模型建模要求，时间序列的预测周期也会极大受到既有数据时间跨度影响



# 文本特征衍生方法
# 我们进一步讨论关于文本字段的特征衍生方法，也被称为NLP（自然语言处理）特征衍生。
# 当然这里所谓的NLP特征衍生并不是真的去处理文本字段，
# 而是借助文本字段的处理方法与思想，来处理数值型的字段，来衍生出更多有效特征。

# 语义分析简例
# 要让计算机来进行情感倾向的识别，我们就需要考虑把这些文本用数值进行表示，
# 例如我们可以用1表示正面情感、0表示负面情感

# 我们如何通过一串数字来描述一段文字呢？最常见的方法就是词袋法，用一个词向量来表示一条文本。
# 词袋法的基本思想是通过对每段文本进行不同单词的计数，然后用这些计数的结果来构成一个词向量，并借此来表示一个文本。

# 有了这个转化我们就将原来的文本语义识别变成了一个二分类问题，
# 而数据集也转化为了结构化数据，其中特征就是每个出现在文本中的单词，
# 而特征的取值就是单词出现的频率。而后我们即可带该数据集进入模型进行训练与预测。

# CountVectorizer与词向量转化
# 词频词向量转化方法

# 很多NLP的情境下下，词频其实就代表着语义的倾向，
# 例如出现了“很好”时（“很好”词频为1）时则表示用户对产品的正面肯定，
# 而出现了“一般”时则表示用户对产品并不满意。
# 此外，如果一句话（也就是一段文本）中重复出现了某些单词，则很有可能表示强调的含义。

# CountVectorizer的sklearn实现
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']

x = vectorizer.fit_transform(corpus)
x.toarray()

# 我们还可以调用vectorizer评估器的get_feature_names属性来查看每一列统计的是哪个单词：
vectorizer.get_feature_names()

c1 = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())


# TF-IDF：词频-逆向词频统计
# 。我们知道，CountVectorizer是简单的单词计数，
# 单词出现的次数越高、我们往往就认为该单词对文本的表意就起到了越关键的作用
# 有的时候简单的词频统计结果无法体现单词对文本表意的重要性。
# 若要降低这些出现频率很高但又没什么用的单词的数值权重，
# 就需要使用另一种词向量转化方法——TF-IDF。

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(c1)

tfidf.toarray()



# NLP特征衍生的简单应用
# 在电信用户数据集中，存在一些彼此“类似”但又相互补充的离散字段，
# 即用户购买服务字段，这些字段记录了用户此前购买的一系列服务，
# 包括是否开通网络安全服务（OnlineSecurity）、是否开通在线备份服务（OnlineBackup）

# CountVectorizer与分组统计特征衍生
# 这里首先将OnlineSecurity、OnlineBackup和DeviceProtection视作文本中的不同单词（Term），
# 并根据tenure不同取值对用户进行分组，每个分组视作一个Document，
# 然后对每个文件进行词频的汇总

# 代码实现
tar_col = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
keycol = 'tenure'
features['OnlineSecurity'].explode().value_counts().to_dict()

# 我们需要将购买了服务标记为1，其他标记为0，可以通过如下操作实现：
features_OE = pd.DataFrame()
features_OE[keycol] = features[keycol]
for col in tar_col:
    features_OE[col] = (features[col] == 'Yes') * 1

# 整体过程并不复杂，和分组汇总统计特征一样，稍后我们将这些特征以tenure为主键拼接回原特征矩阵即可形成衍生特征。