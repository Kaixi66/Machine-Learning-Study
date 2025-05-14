# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt



# 回归数据创建函数
def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程最高项次数
    :return: 生成的特征张和标签张量
    """
    
    if bias == True:
        num_inputs = len(w)-1                                                           # 数据集特征个数
        features_true = np.random.randn(num_examples, num_inputs)                       # 原始特征
        w_true = np.array(w[:-1]).reshape(-1, 1)                                        # 自变量系数
        b_true = np.array(w[-1])                                                        # 截距
        labels_true = np.power(features_true, deg).dot(w_true) + b_true                 # 严格满足人造规律的标签
        features = np.concatenate((features_true, np.ones_like(labels_true)), axis=1)    # 加上全为1的一列之后的特征
    else: 
        num_inputs = len(w)
        features = np.random.randn(num_examples, num_inputs) 
        w_true = np.array(w).reshape(-1, 1)         
        labels_true = np.power(features, deg).dot(w_true)
    labels = labels_true + np.random.normal(size = labels_true.shape) * delta
    return features, labels

# SSE计算函数
def SSELoss(X, w, y):
    """
    SSE计算函数
    
    :param X：输入数据的特征矩阵
    :param w：线性方程参数
    :param y：输入数据的标签数组
    :return SSE：返回对应数据集预测结果和真实结果的误差平方和 
    """
    y_hat = X.dot(w)
    SSE = (y - y_hat).T.dot(y - y_hat)
    return SSE

def MSELoss(X, w, y):
    """
    MSE指标计算函数
    """
    SSE = SSELoss(X, w, y)
    MSE = SSE / X.shape[0]
    return MSE

def lr_gd(X, w, y):
    """
    线性回归梯度计算公式
    """
    m = X.shape[0]
    grad = 2 * X.T.dot((X.dot(w) - y)) / m
    return grad

# 数据集随机切分函数
def array_split(features, labels, rate=0.7, random_state=24):
    """
    训练集和测试集切分函数
    
    :param features: 输入的特征张量
    :param labels：输入的标签张量
    :param rate：训练集占所有数据的比例
    :random_state：随机数种子值
    :return Xtrain, Xtest, ytrain, ytest：返回特征张量的训练集、测试集，以及标签张量的训练集、测试集 
    """
    
    features = np.copy(features)
    labels = np.copy(labels)
    np.random.seed(random_state)                           
    np.random.shuffle(features)                             # 对特征进行切分
    np.random.seed(random_state)
    np.random.shuffle(labels)                               # 按照相同方式对标签进行切分
    num_input = len(labels)                                 # 总数据量
    split_indices = int(num_input * rate)                   # 数据集划分的标记指标
    Xtrain, Xtest = np.vsplit(features, [split_indices, ])  
    ytrain, ytest = np.vsplit(labels, [split_indices, ])
    return Xtrain, Xtest, ytrain, ytest

def sigmoid(x):
    """
    Sigmoid函数
    """
    return (1 / (1 + np.exp(-x)))

def sigmoid_deri(x):
    """
    Sigmoid函数导函数
    """
    return (sigmoid(x)*(1-sigmoid(x)))

def logit_cla(yhat, thr=0.5):
    """
    逻辑回归类别输出函数：
    :param yhat: 模型输出结果
    :param thr：阈值
    :return ycla：类别判别结果
    """
    ycla = np.zeros_like(yhat)
    ycla[yhat >= thr] = 1
    return ycla

def entropy(p):
    """
    信息熵计算函数
    """
    if p == 0 or p == 1:
        ent = 0
    else:
        ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return ent

def dist(x, y, cat = 2):
    """
    闵可夫斯基距离计算函数
    """
    d1 = np.abs(x - y)
    if x.ndim > 1 or y.ndim > 1:
        res1 = np.power(d1, cat).sum(1)
    else:
        res1 = np.power(d1, cat).sum()
    res = np.power(res1, 1/cat)
    return res

# Lesson 4.3案例用函数，非通用函数
def gd(lr = 0.02, itera_times = 20, w = 10):
    """
    梯度下降计算函数
    :param lr: 学习率
    :param itera_times：迭代次数
    :param w：参数初始取值
    :return results：每一轮迭代的参数计算结果列表
    """                              
    results = [w]
    for i in range(itera_times):
        w -= lr * 28 * (w - 2)            # 梯度计算公式
        results.append(w)
    return results

def show_trace(res):
    """
    梯度下降轨迹绘制函数
    """
    f_line = np.arange(-6, 10, 0.1)
    plt.plot(f_line, [14 * np.power(x-2, 2) for x in f_line])
    plt.plot(res, [14 * np.power(x-2, 2) for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('Loss(x)')

def l2d(x):
    """
    二维梯度下降示例梯度计算函数
    """
    a = np.array([[20, 8], [8, 4]])
    b = np.array([28, 12])
    return np.dot(a, x)-b

# Lesson 4.3通用函数
def lr_gd(X, w, y):
    """
    线性回归梯度计算公式
    """
    m = X.shape[0]
    grad = 2 * X.T.dot((X.dot(w) - y)) / m
    return grad

def w_cal(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    """
    梯度下降中参数更新函数 
    :param X: 训练数据特征
    :param w: 初始参数取值
    :param y: 训练数据标签
    :param gd_cal：梯度计算公式
    :param lr: 学习率
    :param itera_times: 迭代次数       
    :return w：最终参数计算结果   
    """
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
    return w

def w_cal_rec(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    """
    在w_cal函数基础上添加梯度记录功能
    """
    w_res = [np.copy(w)]
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
        w_res.append(np.copy(w))
    return w, w_res

def sgd_cal(X, w, y, gd_cal, epoch, batch_size=1, lr=0.02, shuffle=True, random_state=24):
    """
    随机梯度下降和小批量梯度下降计算函数
    :param X: 训练数据特征
    :param w: 初始参数取值
    :param y: 训练数据标签
    :param gd_cal：梯度计算公式
    :param epoch: 遍历数据集次数
    :batch_size: 每一个小批包含数据集的数量
    :param lr: 学习率
    :shuffle：是否在每个epoch开始前对数据集进行乱序处理
    :random_state：随机数种子值
    :return w：最终参数计算结果       
    """
    m = X.shape[0]
    n = X.shape[1]
    batch_num = np.ceil(m / batch_size)
    X = np.copy(X)
    y = np.copy(y)
    for j in range(epoch):
        if shuffle:
            np.random.seed(random_state)         
            np.random.shuffle(X)                 
            np.random.seed(random_state)
            np.random.shuffle(y)    
        for i in range(np.int(batch_num)):
            w = w_cal(X[i*batch_size: np.min([(i+1)*batch_size, m])], 
                      w, 
                      y[i*batch_size: np.min([(i+1)*batch_size, m])], 
                      gd_cal=gd_cal, 
                      lr=lr, 
                      itera_times = 1)
    return w

def z_score(X):
    """
    Z-Score标准化函数
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)


def maxmin_norm(X):
    """
    max—min normalization标准化函数
    """
    maxmin_range = X.max(axis=0) - X.min(axis=0)
    return (X - X.min(axis=0)) / maxmin_range


def logit_gd(X, w, y):
    """
    逻辑回归梯度计算公式
    """
    m = X.shape[0]
    grad = X.T.dot(sigmoid(X.dot(w)) - y) / m
    return grad

def arrayGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """分类数据集创建函数。
    
    :param num_examples: 每个类别的数据数量
    :param num_inputs: 数据集特征数量
    :param num_class：数据集标签类别总数
    :param deg_dispersion：数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
    :param bias：建立模型逻辑回归模型时是否带入截距，为True时将添加一列取值全为1的列
    :return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组。
    """
    
    cluster_l = np.empty([num_examples, 1])                            # 每一类标签数组的形状
    mean_ = deg_dispersion[0]                                        # 每一类特征数组的均值的参考值
    std_ = deg_dispersion[1]                                         # 每一类特征数组的方差
    lf = []                                                          # 用于存储每一类特征的列表容器
    ll = []                                                          # 用于存储每一类标签的列表容器
    k = mean_ * (num_class-1) / 2                                    # 每一类特征均值的惩罚因子
    
    for i in range(num_class):
        data_temp = np.random.normal(i*mean_-k, std_, size=(num_examples, num_inputs))     # 生成每一类特征
        lf.append(data_temp)                                                               # 将每一类特征添加到lf中
        labels_temp = np.full_like(cluster_l, i)                                           # 生成某一类的标签
        ll.append(labels_temp)                                                             # 将每一类标签添加到ll中
        
    features = np.concatenate(lf)
    labels = np.concatenate(ll)
    
    if bias == True:
        features = np.concatenate((features, np.ones(labels.shape)), 1)    # 在特征张量中添加一列全是1的列
    return features, labels

def logit_acc(X, w, y, thr=0.5):
    """
    逻辑回归准确率计算函数
    """
    yhat = sigmoid(X.dot(w))
    y_cal = logit_cla(yhat, thr=thr)
    return (y_cal== y).mean()

def logit_DB(X, w, y):
    """
    逻辑回归决策边界绘制函数
    """
    
    # 以两个特征的极值+1/-1作为边界，并在其中添加1000个点
    x1, x2 = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 1000).reshape(-1,1),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 1000).reshape(-1,1))
    
    # 将所有点的横纵坐标转化成二维数组
    X_temp = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1), np.ones(shape=(1000000, 1))], 1)
    
    # 对所有点进行逻辑回归预测
    y_hat_temp = logit_cla(sigmoid(X_temp.dot(w)))
    yhat = y_hat_temp.reshape(x1.shape)
    
    # 绘制决策边界图像
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#90CAF9'])
    plt.contourf(x1, x2, yhat, cmap=custom_cmap)
    
    