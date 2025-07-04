# 1. 业务背景解读与数据探索
# 第一时间，需要对数据（也就是对应业务）的基本背景进行解读,
# 数据探索包括数据分布检验、数据正确性校验、数据质量检验、训练集/测试集规律一致性检验等。

# 2. 数据预处理与特征工程
# 数据清洗主要聚焦于数据集数据质量提升，包括缺失值、异常值、重复值处理，以及数据字段类型调整等；
# 而特征工程部分则更倾向于调整特征基本结构，来使数据集本身规律更容易被模型识别，
# 如特征衍生、特殊类型字段处理（包括时序字段、文本字段等）等。

# 3. 算法建模与模型调优
# 建模过程既包括算法训练也包括参数调优


# 业务背景与数据背景
# 本次案例的数据源自Kaggle平台上分享的建模数据集：Telco Customer Churn，
# 该数据集描述了某电信公司的用户基本情况，包括每位用户已注册的相关服务、用户账户信息、用户人口统计信息等，
# 当然，也包括了最为核心的、也是后续建模需要预测的标签字段——用户流失情况（Churn）。

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 读取数据
tcc = pd.read_csv(r'D:\pythonPractice\practice_example\TelcoCustomerChurn\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 控制表格列宽的最大显示长度。
pd.set_option('max_colwidth', 200)
print(tcc.head())
tcc.info()

# Discussion页面中的讨论帖都是重要的信息获取渠道

#实际的算法建模目标有两个，
# 其一是对流失用户进行预测，
# 其二则是找出影响用户流失的重要因子，来辅助运营人员来进行营销策略调整或制定用户挽留措施。

# 综合上述两个目标我们不难发现，我们要求模型不仅要拥有一定的预测能力，并且能够输出相应的特征重要性排名，
# 并且最好能够具备一定的可解释性，也就是能够较为明显的阐述特征变化是如何影响标签取值变化的
# 逻辑回归的线性方程能够提供非常好的结果可解释性，同时我们也可以通过逻辑回归中的正则化项也可以用于评估特征重要性。


# 数据解读与预处理
# 根据官方给出的数据集说明，上述字段基本可以分为三类，分别是用户已注册的服务信息、用户账户信息和用户人口统计信息，三类字段划分情况如下：


#数据质量探索
# 数据集正确性校验
# 一般来说数据集正确性校验分为两种，
# 其一是检验数据集字段是否和数据字典中的字段一致，其二则是检验数据集中ID列有无重复。
# 由于该数据集并未提供数据字典，因此此处主要校验数据集ID有无重复：
print(tcc['customerID'].nunique() == tcc.shape[0])

# ID列没有重复，则数据集中也不存在完全重复的两行数据：
print(tcc.duplicated().sum())


# 数据缺失值检验
print(tcc.isnull().sum())

# 此外，我们也可以通过定义如下函数来输出更加完整的每一列缺失值的数值和占比：
def missing (df):
    """
    计算每一列的缺失值及占比
    """
    missing_number = df.isnull().sum().sort_values(ascending=False)              # 每一列的缺失值求和后降序排序                  
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)          # 每一列缺失值占比
    # .count()：这里统计的是每列中非缺失值的数量
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])      # 合并为一个DataFrame
    return missing_values

# 在info返回的信息中的non-null也能看出数据集不存在缺失值。
# 没有缺失只代表数据集中没有None或者Nan，并不排除可能存在用别的值表示缺失值的情况，稍后我们将对其进行进一步分析。


# 字段类型探索
# 时序字段处理
# 根据数据集info我们发现，大多数字段都属于离散型字段，并且object类型居多。
# 我们是无法直接使用object类型对象的，因此需要对其进行类型转化，
# 通常来说，我们会将字段划分为连续型字段和离散型字段，并且根据离散字段的具体含义来进一步区分是名义型变量还是有序变量。
# 名义型变量:仅分类，无顺序
# 有序变量:有明确的等级或顺序关系

# 不过在划分连续/离散字段之前，我们发现数据集中存在一个入网时间字段，看起来像是时序字段
# 从严格意义上来说，用时间标注的时序字段即不数据连续型字段或离散型字段（尽管可以将其看成是离散字段，但这样做会损失一些信息），
# 因此我们需要重点关注入网时间字段是否是时间标注的字段：

print(tcc['tenure'])
# 该字段并不是典型的用年月日标注的时间字段，如2020-08-01，而是一串连续的数值。当然，我们可以进一步查看该字段的取值范围：
print(tcc['tenure'].nunique())

# 该字段总共有73个不同的取值，结合此前所说，数据集是第三季度的用户数据，因此我们推断该字段应该是经过字典排序后的离散型字段。
# 字典排序，其本质是一种离散变量的转化方式，有时我们也可以将时序数据进行字典排序
# 总共有4个不同时间，视作时间字段有4个不同取值。再将四个不同取值重新编码，根据排序结果将其编码为1/2/3.../n

# 我们可以转化后的入网时间字段看成是离散变量，当然也可以将其视作连续变量来进行分析，具体选择需要依据模型来决定。
# 此处我们先将其视作离散变量，后续根据情况来进行调整。

# 接下来，我们来标注每一列的数据类型，我们可以通过不同列表来存储不同类型字段的名称：
# 离散字段
category_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod']

# 连续字段
numeric_cols = ['MonthlyCharges', 'TotalCharges']

# 标签
target = 'Churn'

# 验证是否划分能完全
assert len(category_cols) + len(numeric_cols) + 1 == tcc.shape[1]

# 大多数时候离散型字段都在读取时都是object类型，因此我们也可以通过如下方式直接提取object字段：
print(tcc.select_dtypes('object').columns) # .columns获取列名称

# 然后，我们需要对不同类型字段进行转化。并且在此过程中，我们需要检验是否存在采用别的值来表示缺失值的情况。
# 就像此前所说我们通过isnull只能检验出None(Python原生对象)和np.Nan(numpy/pandas在读取数据文件时文件内部缺失对象的读取后表示形式)对象。
# 但此外我们还需要注意数据集中是否包含采用某符号表示缺失值的情况，
# 例如某些时候可能使用空格（其本质也是一种字符）来代替空格
print(tcc[category_cols].nunique())

# 我们也可以通过如下方式查看每个离散变量的不同取值：
for feature in tcc[category_cols]:
    print(f'{feature}:{tcc[feature].unique()}')

# 通过对比离散变量的取值水平，我们发现并不存在通过其他值表示缺失值的情况。
# 如果是连续变量，则无法使用上述方法进行检验（取值水平较多），
# 但由于往往我们需要将其转化为数值型变量再进行分析，因此对于连续变量是否存在其他值表示缺失值的情况，
# 我们也可以观察转化情况来判别，例如如果是用空格代表缺失值，则无法直接使用astype来转化成数值类型。

# 缺失值检验与填补
#  发现在连续特征中存在空格。
# 则此时我们需要进一步检查空格字符出现在哪一列的哪个位置，我们可以通过如下函数来进行检验：

def find_index(data_col, val):
    """
    查询某值在某列中第一次出现位置的索引，没有则返回-1
    
    :param data_col: 查询的列
    :param val: 具体取值
    """
    val_list = [val]
    if data_col.isin(val_list).sum() == 0:
        index = -1
    else:
        index = data_col.isin(val_list).idxmax()
    return index

for col in numeric_cols:
    print(find_index(tcc[col], ' '))

# 即空格第一次出现在'TotalCharges'列的索引值为488的位置：
# 接下来使用np.nan对空格进行替换，并将'MonthlyCharges'转化为浮点数类型：
tcc['TotalCharges']= tcc['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)

# 关于该缺失值应该如何填补，首先考虑的是，由于缺失值占比较小，因此可以直接使用均值进行填充：
tcc['TotalCharges'].fillna(tcc['TotalCharges'].mean())

# 此外，我们也可以简单观察缺失'TotalCharges'信息的每条数据实际情况，或许能发现一些蛛丝马迹：
print(tcc[tcc['TotalCharges'].isnull()]) 
# isnull 返回的是行数，这里查看totalcharges为nan的行的情况

# 我们发现，这11条数据的入网时间都是0，也就是说，这11位用户极有可能是在统计周期结束前的最后时间入网的用户，
# 因此没有过去的总消费记录，但是却有当月的消费记录。
# 也就是说，该数据集的过去总消费记录不包括当月消费记录，也就是不存在过去总消费记录等于0的记录。我们可以简单验证：
print((tcc['TotalCharges'] == 0).sum()) # .sum()统计true的个数

# 既然如此，我们就可以将这11条记录的缺失值记录为0，以表示在最后一个月统计消费金额前，这些用户的过去总消费金额为0:
tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)
print(tcc['TotalCharges'].isnull().sum())


# 此外还有另一种便捷的方式，即直接使用pd.to_numeric对连续变量进行转化，
# 并在errors参数位上输入'coerce'参数，表示能直接转化为数值类型时直接转化，无法转化的值(空格，字符串)用缺失值填补，过程如下：
df1 = pd.read_csv(r'D:\pythonPractice\practice_example\TelcoCustomerChurn\WA_Fn-UseC_-Telco-Customer-Churn.csv')
df1.TotalCharges = pd.to_numeric(df1.TotalCharges, errors='coerce')

# 查看空格处是否被标记为缺失值
print(df1.TotalCharges.iloc[488])
print(df1.TotalCharges.dtype)

# 此处暂时未对离散特征进行变量类型转化，
# 是因为本小节后半段需要围绕标签取值在不同特征维度上分布进行分析，
# 此时需要查看各特征的原始取值情况（例如性别是Male和Female，而不是0/1），
# 因此我们会在本节结束后对离散变量再进行字典编码。

# 异常值检测
# 对于连续型变量，我们可以进一步对其进行异常值检测。首先我们可以采用describe方法整体查看连续变量基本统计结果：
print(tcc[numeric_cols].describe())

# 我们可以通过三倍标准差法来进行检验，
# 即以均值-3倍标注差为下界，均值+3倍标准差为上界，来检测是否有超过边界的点：
for col in numeric_cols:
    if tcc[col].max() > tcc[col].mean() + 3 * tcc[col].std() or tcc[col].min() < tcc[col].mean() - 3 * tcc[col].std():
        print('有异常值')
    print(f'{col}:无异常值')

# 对于异常值的检测和处理也是需要根据实际数据分布和业务情况来判定，
# 一般来说，数据分布越倾向于正态分布，则通过三倍标准差或者箱线图检测的异常值会更加准确一些

# 变量相关性探索分析与探索性分析
# 接下来我们可以通过探索标签在不同特征上的分布，来初步探索哪些特征对标签取值影响较大。
# 当然，首先我们可以先查看标签字段的取值分布情况：

y = tcc['Churn']
print(f'Percentage of Churn:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} customer)\nPercentage of customer did not churn: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} customer)')

# 就是在总共7000余条数据中，流失用户占比约为26%，整体来看标签取值并不均匀，
# 但如果放到用户流失这一实际业务背景中来探讨，流失用户比例占比26%已经是非常高的情况了。
# 当然我们也可以通过直方图进行直观的观察：
sns.displot(y)
#plt.show()  # 强制显示图形


# 变量相关性分析
# 不同类型变量的相关性需要采用不同的分析方法:
# 连续变量之间相关性可以使用皮尔逊相关系数进行计算，
# 连续变量和离散变量之间相关性则可以卡方检验进行分析，
# 离散变量之间则可以从信息增益角度入手进行分析

# 但是，如果我们只是想初步探查变量之间是否存在相关关系，则可以忽略变量连续/离散特性，统一使用相关系数进行计算，
# 这也是pandas中的.corr方法所采用的策略。

# 计算相关系数矩阵
# 直接通过具体数值大小来表示相关性强弱。
# 不过需要注意的是，尽管我们可以忽略变量的连续/离散特性，
# 但为了更好的分析分类变量如何影响标签的取值，
# 我们需要将标签转化为整型（也就是视作连续变量），
# 而将所有的分类变量进行哑变量处理： 通过独热编码（One-Hot Encoding 将分类特征转换为数值特征

# 剔除ID列
df3 = tcc.iloc[:, 1:].copy()

# 将标签yes/no转化为1/0
df3['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df3['Churn'].replace(to_replace='No', value=0, inplace=True)
# 将其他所有分类变量转化为哑变量，连续变量保留不变
# pd.get_dummies会将非数值类型对象类型进行自动哑变量转化，而对数值类型对象，无论是整型还是浮点型，都会保留原始列不变：
df_dummies = pd.get_dummies(df3)

# 然后即可采用.corr方法计算相关系数矩阵：
print(df_dummies.corr())
# 在所有的相关性中，我们较为关注特征和标签之间的相关关系，
# 因此可以直接挑选标签列的相关系数计算结果，并进行降序排序：
print(df_dummies.corr()['Churn'].sort_values(ascending = False))

# 根据相关系数计算的基本原理，相关系数为正数，则二者为正相关，数值变化会更倾向于保持同步。
# 例如Churn与Contract_Month-to-month相关系数为0.4，则说明二者存在一定的正相关性
# 而tenure和Churn负相关，则说明tenure取值越大、用户流失概率越小

# 柱状图
sns.set()
plt.figure(figsize=(15, 8), dpi=200)

df_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.show()

# 探索性数据分析
# 时我们可以考虑围绕不同类型的属性进行柱状图的展示与分析。当然，此处需要对比不同字段不同取值下流失用户的占比情况，
# 因此可以考虑使用柱状图的另一种变形：堆叠柱状图来进行可视化展示：
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,6), dpi=100)

# 柱状图
plt.subplot(121)
sns.countplot(x="gender",hue="Churn",data=tcc,palette="Blues", dodge=True)
plt.xlabel("Gender")
plt.title("Churn by Gender")

# 柱状堆叠图
plt.subplot(122)
sns.countplot(x="gender",hue="Churn",data=tcc,palette="Blues", dodge=False)
plt.xlabel("Gender")
plt.title("Churn by Gender")