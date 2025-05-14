# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 自定义模块
from ML_basic_function import *

# 由于相对熵=交叉熵-信息熵，因此我们只能力求减少交叉熵。
# 当然，也正因如此，交叉熵可以作为衡量模型输出分布是否接近真实分布的重要度量方法。

# 二分类交叉熵损失函数
def BCE(y, yhat):
    return (-(1/len(y))) * np.sum(y * np.log2(yhat) + (1 - y) * np.log2(1 - yhat))

y = np.array([1, 0, 0, 1]).reshape(-1, 1)
yhat = np.array([0.8, 0.3, 0.4, 0.7]).reshape(-1, 1)
print(BCE(y, yhat))