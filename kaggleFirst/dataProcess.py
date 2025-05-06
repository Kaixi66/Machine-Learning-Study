import os
import numpy as np
import pandas as pd
import gc
import seaborn as sns
import matplotlib.pyplot as plt
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
sns.set()
sns.histplot(train['target'], kde=True)
plt.show()   
print((train['target'] < -30).sum())