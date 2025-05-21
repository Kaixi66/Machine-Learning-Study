# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

# Scikit-Learn相关模块
# 评估器类
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 实用函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 来执行包含特征衍生和正则化过程的建模试验
# 本节的实验将为下一小节的网格搜索调参做铺垫

# 数据准备
# 我们可以创建一个满足分类边界为 y**2 = -x + 1.5 的分布，创建方法如下：
np.random.seed(24)
X = np.random.normal(0, 1, size=(1000, 2)) # 标准正态分布
y = np.array(X[:, 0] + X[:, 1] ** 2 < 1.5, int)
#plt.scatter(X[:, 0], X[:, 1], c=y)


# 加上扰动项

np.random.seed(24)
for i in range(200):
    y[np.random.randint(1000)] = 1
    y[np.random.randint(1000)] = 0
#lt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
# 数据准备完毕

#构建机器学习流
# 1. 借助此前介绍的PolynomialFeatures来进行特征衍生
# 2. 对数据进行标准化处理
# 3. 通过Pipeline将这些过程封装在一个机器学习流中
# 4. 如果出现过拟合之后应该如何调整

# 封装在一个函数中
# degree：多项式的阶数， penalty： 是否正则化， C：值越低，正则化越强， tol：优化算法的收敛阈值。当损失函数的变化小于该值时，停止迭代。，越小精度越高

def plr(degree=1, penalty='none', C=1.0):
    pipe = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                        StandardScaler(),
                        LogisticRegression(penalty=penalty, tol=1e-4, C=C, max_iter=int(1e6)))
    return pipe


# 评估器训练与过拟合实验
# 接下来进行模型训练，并且尝试进行手动调参来控制模型拟合度。
pl1 = plr()
# get_params() 是获取模型或流水线完整配置的便捷工具
# 调整polynomialFeatures评估器
#这里的polynomialfeatures步骤的名称是根据类名自动生成的（小写 + 下划线）。
# 在流水线中修改组件参数时，使用 步骤名__参数名 的语法
# 仅修改当前实例（如 pl1），不影响原函数 plr()
pl1.set_params(polynomialfeatures__include_bias=True)
print(pl1.get_params()['polynomialfeatures__include_bias'])

# 测试模型性能，首先是不进行特征衍生的逻辑回归建模结果,只有一次项特征
pr1 = plr()
pr1.fit(X_train, y_train)
print(pr1.score(X_train, y_train), pr1.score(X_test, y_test))

# 直观的模型建模结果的观察
def plot_decision_boundary(X, y, model):
    """
    决策边界绘制函数
    """
    
    # 以两个特征的极值+1/-1作为边界，并在其中添加1000个点
    x1, x2 = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 1000).reshape(-1,1),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 1000).reshape(-1,1))
    
    # 将所有点的横纵坐标转化成二维数组
    X_temp = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], 1)
    
    # 对所有点进行模型类别预测
    yhat_temp = model.predict(X_temp)
    yhat = yhat_temp.reshape(x1.shape)
    
    # 绘制决策边界图像
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#90CAF9'])
    plt.contourf(x1, x2, yhat, cmap=custom_cmap)
    plt.scatter(X[(y == 0).flatten(), 0], X[(y == 0).flatten(), 1], color='red')
    plt.scatter(X[(y == 1).flatten(), 0], X[(y == 1).flatten(), 1], color='blue')
    plt.show()
# 测试函数性能
# plot_decision_boundary(X, y, pr1)
# 逻辑回归在不进行数据衍生的情况下，只能捕捉线性边界.
# 我们尝试衍生2次项特征再来进行建模
pr2 = plr(degree=2)
pr2.fit(X_train, y_train)
print(pr2.score(X_train, y_train), pr2.score(X_test, y_test))
# plot_decision_boundary(X, y, pr2)
# 可以发现，模型效果有明显提升，验证当前数据特征数量：
# named_steps返回结果同样也是一个字典，通过key来调用对应评估器。
print(pr2.named_steps)
print(pr2.named_steps['logisticregression'].coef_)
# 对应训练数据总共5个特征，说明最高次方为二次方、并且存在交叉项目的特征衍生顺利执行。
# 在进行特征衍生的时候，就相当于是将原始数据集投射到一个高维空间
# 突破了逻辑回归在原始特征空间中的线性边界的束缚
# 经过特征衍生的逻辑回归模型，也将在原始特征空间中呈现出非线性的决策边界的特性。
# 无论是几阶的特征衍生，能够投射到的高维空间都是有限的，而我们最终也只能在这些有限的高维空间中寻找一个最优的超平面。

# 过拟合倾向实验
# 我们可以进一步进行10阶特征的衍生，然后构建一个更加复杂的模型
pr3 = plr(degree=10)
pr3.fit(X_train, y_train)
print(pr3.score(X_train, y_train), pr3.score(X_test, y_test))
# 入股在运行过程受到警告，迭代次数（max_iter）用尽，但并没有收敛到tol参数设置的区间
# 三种解决方法：1. 增加max_iter迭代次数，2.增加收敛区间，3.加入正则化
# 但此处由于我们本身只设置了1000条数据，较小的数据量是目前无法收敛止较小区间的根本原因

# 要求改tol参数，则可以使用前面介绍的set_param方法来进行修改：
print(pr3.get_params())
pr3 = plr(degree=10)
pr3.set_params(logisticregression__tol=1e-2)
pr3.fit(X_train, y_train)
print(pr3.score(X_train, y_train), pr3.score(X_test, y_test))

#放宽了收敛条件
# 不难看出，模型已呈现出过拟合倾向。
# plot_decision_boundary(X, y, pr3)

# 接下来我们可以尝试通过衍生更高阶特征来提高模型复杂度
score_l = []
# 实例化多组模型，测试模型效果
for degree in range(1, 21):
    pr_temp = plr(degree=degree)
    pr_temp.fit(X_train, y_train)
    score_temp = [pr_temp.score(X_train, y_train), pr_temp.score(X_test, y_test)]
    score_l.append(score_temp)
arr = np.array(score_l)
print(arr)
'''
plt.plot(list(range(1, 21)), np.array(score_l)[:,0], label='train_acc')
plt.plot(list(range(1, 21)), np.array(score_l)[:,1], label='test_acc')
plt.legend(loc = 4)
plt.show()
'''
#最终，我们能够较为明显的看出，伴随着模型越来越复杂（特征越来越多），训练集准确率逐渐提升，
# 但测试集准确率却在一段时间后开始下降，说明模型经历了由开始的欠拟合到拟合再到过拟合的过程，
# 和上一小节介绍的模型结构风险伴随模型复杂度提升而提升的结论一致。

# 采用正则化将是一个不错的选择，当然我们也可以直接从上图中的曲线变化情况来挑选最佳的特征衍生个数。
# 建模流程是先进行数据增强（特征衍生），来提升模型表现，然后再通过正则化的方式来抑制过拟合倾向。

# 验证正则化对过拟合的抑制效果
pl1 = plr(degree=10, penalty='l1', C=1.0)
# 更改求解器，支持更灵活的正则化，提高大规模数据效率
pl1.set_params(logisticregression__solver='saga') 
pl1.fit(X_train, y_train)
print(pl1.score(X_train, y_train), pl1.score(X_test, y_test))

# 测试l2正则化
pl2 = plr(degree=10, penalty='l2', C=1.0).fit(X_train, y_train)
print(pl2.score(X_train, y_train), pl2.score(X_test, y_test))
plot_decision_boundary(X, y, pl2)
