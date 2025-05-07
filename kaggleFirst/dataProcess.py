import os
import numpy as np
import pandas as pd
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import time
dictionPath = r"D:\kaggleData\competitions\Data_Dictionary.xlsx"
table = pd.read_excel(dictionPath, header=2, sheet_name='train')
#print(table)

samplePath = r"D:\kaggleData\competitions\sample_submission.csv"
sampleTable = pd.read_csv(samplePath, header=0).head(5)
# nsampleTable = pd.read_csv(samplePath, header=0).info()
#print(sampleTable)

train = pd.read_csv(r"D:\kaggleData\competitions\train.csv")
test = pd.read_csv(r"D:\kaggleData\competitions\test.csv")
#print(train.shape, test.shape)
#print(train['card_id'])

# testify data quality
#print(train['card_id'].nunique() == train.shape[0])
#print(test['card_id'].nunique() == test.shape[0])
'''
nunique()是 pandas 中 Series 对象的一个方法。
作用是统计该 Series 中唯一值的数量。
shape[0] refers to the number of rows, [1] refers to the number of column
'''

# check null value
#print(train.isnull().sum()) # sum up null value base on column
#print(test.isnull().sum())

#检测异常值,发现有异常值在-30附件，我们可以看多少用户值是低于30,一共2207
'''
sns.set()
sns.histplot(train['target'], kde=True)
plt.show()   
print((train['target'] < -30).sum())
'''
#单变量分析test和train规律是否一致
#因为data中四个变量都是离散型变量，因此分布规律可以通过相对占比分布进行比较，每个active month的人数占总人数的百分之多少
features = ['first_active_month', 'feature_1', 'feature_2','feature_3']

#样本总数
train_count = train.shape[0]
test_count = test.shape[0]

# .value_counts()是 pandas 中 Series 对象（这里 'first_active_month' 列就是一个 Series）的方法。它的作用是统计该列中每个不同值出现的次数。
'''
for feature in features:
    (train[feature].value_counts().sort_index()/train_count).plot()
    (test[feature].value_counts().sort_index()/test_count).plot()
    plt.legend(['train', 'test'])
    plt.xlabel(feature)
    plt.ylabel('ratio')
    plt.show()
'''

# 对merchant数据进行分析
merchant = pd.read_csv(r"D:\kaggleData\competitions\merchants.csv")
'''
print(merchant.shape, merchant['merchant_id'].nunique())
print(merchant.isnull().sum())
'''
#划分离散和连续变量
category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']

# 检验特征是否划分完全
assert len(category_cols) + len(numeric_cols) == merchant.shape[1] # 维度为1代表列数，也就是每个series的标签
#assert 语句用于调试，它会对一个条件进行检查，如果条件为 False，则会抛出一个 AssertionError 异常，从而终止程序的执行。该语句的目的是确保某个条件为真，以保证程序后续的操作能够正常进行。
'''''
print(merchant[category_cols].dtypes) # 查看category类型数据的数据类型
print(merchant[category_cols].isnull().sum()) #可以看到category_2缺失值较多，需要进行填补
print(merchant['category_2'].unique()) # 发现这个series中有缺失值nan
'''''
merchant['category_2'] = merchant['category_2'].fillna(-1) # 把缺失值转化为-1

#字母编码函数 把数据中的字母转换为数字
def change_object_cols(se):
    value =se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values
    # 创建一个新的 Series，其索引为排序后的唯一值，值为对应的数字编码
    # 这里使用 pd.Series(range(len(value)), index=value) 创建了一个映射关系, index = value指的是把value中元素都变成key来使用
    # 例如，如果 value 是 ['a', 'b', 'c']，则创建的 Series 为 {'a': 0, 'b': 1, 'c': 2}
    # 然后使用 map 方法将原 Series 中的每个值替换为对应的数字编码
    # 最后使用 values 属性将结果转换为 NumPy 数组

# 接下来，对merchant对象中的四个object类型列进行类别转化：
for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])


# 查看连续变量整体情况
'''
print(merchant[numeric_cols].isnull().sum()) # 此处发现有3个类型缺失值为13
print(merchant[numeric_cols].describe()) #部分连续变量还存在无穷值inf，需要对其进行简单处理。
'''
#此处我们首先需要对无穷值进行处理。此处我们采用类似天花板盖帽法的方式对其进行修改，即将inf改为最大的显式数值。
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())
# 它会先把正无穷大值替换成 -99，然后找出这些列在替换后数据中的最大值，最后再用这个最大值去替换原始数据里的正无穷大值。这里第二个 max() 是对前面得到的 Series 对象再求一次最大值，最终得到的是替换后所有列中的全局最大值。

#缺失值处理
#不同于无穷值的处理，缺失值处理方法有很多。但该数据集缺失数据较少，33万条数据中只有13条连续特征缺失值，此处我们先简单采用均值进行填补处理，后续若有需要再进行优化处理。
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())

# 6、去除与transaction交易记录表格重复的列，以及merchant_id的重复记录。
duplicate_cols = ['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id', 'category_2']
merchant = merchant.drop(duplicate_cols[1:], axis=1)
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)
'''
print("-------------------------------")
print(merchant[numeric_cols].describe())
'''

# transaction数据处理
history_transaction = pd.read_csv(r"D:\kaggleData\competitions\historical_transactions.csv", header=0)
label = pd.read_excel(r"D:\kaggleData\competitions\Data_Dictionary.xlsx", header=2, sheet_name='history') # 查询字典找到分类名字解释
new_transaction = pd.read_csv(r"D:\kaggleData\competitions\new_merchant_transactions.csv", header=0)

#因为之后merchant和transaction文件要合并，所以先查重
duplicate_cols = [] 
for col in merchant.columns:
    if col in new_transaction.columns:
        duplicate_cols.append(col)
        
# 取出和商户数据表重复字段并去重
# print(new_transaction[duplicate_cols].drop_duplicates().shape)
# print(new_transaction['merchant_id'].nunique())

numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
       'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
       'subsector_id']
time_cols = ['purchase_date']

assert len(numeric_cols) + len(category_cols) + len(time_cols) == new_transaction.shape[1]

#print(new_transaction[category_cols].isnull().sum()) #查看缺失值情况
#我们对其object类型对象进行字典编码（id除外），并对利用-1对缺失值进行填补：
for col in ['authorized_flag', 'category_1', 'category_3']:
    new_transaction[col] = change_object_cols(new_transaction[col].fillna(-1).astype(str))

# 再处理不是object类型的缺失值
new_transaction[category_cols] = new_transaction[category_cols].fillna(-1)




#因为之后merchant和transaction文件要合并，所以先查重
duplicate_cols = [] 
for col in merchant.columns:
    if col in history_transaction.columns:
        duplicate_cols.append(col)
        
# 取出和商户数据表重复字段并去重
#print(history_transaction[duplicate_cols].drop_duplicates().shape)
#print(history_transaction['merchant_id'].nunique())

numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
       'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
       'subsector_id']
time_cols = ['purchase_date']

assert len(numeric_cols) + len(category_cols) + len(time_cols) == history_transaction.shape[1]

# print(history_transaction[category_cols].isnull().sum()) #查看缺失值情况
#我们对其object类型对象进行字典编码（id除外），并对利用-1对缺失值进行填补：
for col in ['authorized_flag', 'category_1', 'category_3']:
    history_transaction[col] = change_object_cols(history_transaction[col].fillna(-1).astype(str))

# 再处理不是object类型的缺失值
history_transaction[category_cols] = history_transaction[category_cols].fillna(-1)
# print(history_transaction[category_cols].isnull().sum())

# 对train和test数据进行处理，只用对首次活跃月份进行编码
train = pd.read_csv(r"D:\kaggleData\competitions\train.csv")
test =  pd.read_csv(r"D:\kaggleData\competitions\test.csv")
se_map= change_object_cols(train['first_active_month'].append(test['first_active_month']).astype(str))
train['first_active_month'] = se_map[:train.shape[0]]
test['first_active_month'] = se_map[train.shape[0]:]
'''
train.to_csv("D:/kaggleData/competitions/train_pre.csv", index=False)
test.to_csv("D:/kaggleData/competitions/test_pre.csv", index=False)
'''
#del train 这行代码会把变量 train 对其所指向对象的引用删除。也就是说，现在 train 这个名字不再和那个对象关联。不过要注意，这并不意味着对象本身马上就会从内存中被删除，Python 的垃圾回收机制会在合适的时候处理它。
#gc.collect() 是 gc 模块的一个函数，它的作用是手动触发 Python 的垃圾回收机制。在 Python 中，垃圾回收机制会自动识别那些没有任何引用指向的对象，然后回收它们所占用的内存。
del train
del test
gc.collect()