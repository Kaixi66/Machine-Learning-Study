import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def missing (df):
    """
    计算每一列的缺失值及占比
    """
    missing_number = df.isnull().sum().sort_values(ascending=False)              # 每一列的缺失值求和后降序排序                  
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)          # 每一列缺失值占比
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])      # 合并为一个DataFrame
    return missing_values

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


def colName(ColumnTransformer, numeric_cols, category_cols):
    col_name = []
    col_value = ColumnTransformer.named_transformers_['cat'].categories_
    
    for i, j in enumerate(category_cols):
        if len(col_value[i]) == 2:
            col_name.append(j)
        else:
            for f in col_value[i]:
                feature_name = j + '_' + f
                col_name.append(feature_name)
    col_name.extend(numeric_cols)
    return(col_name)

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

class logit_threshold(BaseEstimator, TransformerMixin):
    
    def __init__(self, penalty='l2', C=1.0, max_iter=1e8, solver='lbfgs', l1_ratio=None, class_weight=None, thr=0.5):
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.thr = thr
        self.class_weight = class_weight
        
    def fit(self, X, y):
        clf = LogisticRegression(penalty = self.penalty, 
                                 C = self.C, 
                                 solver = self.solver, 
                                 l1_ratio = self.l1_ratio,
                                 class_weight=self.class_weight, 
                                 max_iter=self.max_iter)
        clf.fit(X, y)
        self.clf=clf
        return self
        
    def predict(self, X):
        res = (self.clf.predict_proba(X)[:, 1]>=self.thr) * 1
        return res



def Cross_Combination(colSet, df):
    newDf_l = []
    col_name_l = []
    
    for col in colSet:
        for col_sub in colSet:
            if col == col_sub:
                continue
            else:
                col_name = col+'&'+col_sub
                newDf_l.append(pd.Series(df[col].astype('str')+'&'+df[col_sub].astype('str'), name=col_name))
                col_name_l.append(col_name)
    
    newDF = pd.concat(newDf_l, axis=1)
    return newDF, col_name_l


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
                feature_name = str(j) + '_' + str(f)
                cate_cols_new.append(feature_name)
    return(cate_cols_new)