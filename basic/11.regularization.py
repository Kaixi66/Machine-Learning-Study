# 科学计算模块
import numpy as np
import pandas as pd

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# 机器学习模块
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置随机种子，确保结果可复现
np.random.seed(123)

# 生成数据
n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1

# 多项式拟合函数（修正版）
def plot_polynomial_fit(x, y, deg):
    p = np.poly1d(np.polyfit(x, y, deg))
    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'ro', label='Data Points')  # 数据点
    plt.plot(t, p(t), '-', label=f'Poly Fit (deg={deg})')  # 拟合曲线
    plt.plot(t, np.sqrt(t), 'r--', label='True Function')  # 真实函数
    plt.legend()  # 添加图例
    plt.grid(True)  # 添加网格

# 绘制不同阶数的拟合效果
plt.figure(figsize=(18, 4), dpi=200)
titles = ['Under Fitting', 'Good Fitting', 'Over Fitting']
for index, deg in enumerate([1, 3, 10]):
    plt.subplot(1, 3, index + 1)
    plot_polynomial_fit(x, y, deg)
    plt.title(titles[index], fontsize=16)

plt.tight_layout()  # 自动调整布局
plt.show()

# 使用PolynomialFeatures进行特征衍生（修正版）
X = x.reshape(-1, 1)  # 先将x重塑为二维数组
poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.3, random_state=42)

# 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 评估模型
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(f"训练集MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"测试集MSE: {mean_squared_error(y_test, y_test_pred):.4f}")

# 可视化过拟合效果
plt.figure(figsize=(12, 6), dpi=150)
t = np.linspace(0, 1, 200).reshape(-1, 1)
t_poly = poly.transform(t)
plt.plot(x, y, 'ro', label='Data Points')
plt.plot(t, lr.predict(t_poly), '-', label='Model Prediction')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('Overfitting Example (Degree=10 Polynomial)', fontsize=16)
plt.legend()
plt.grid(True)
#plt.show()


#我们尝试在线性回归的损失函数中引入正则化
# 导入岭回归和Lasso
from sklearn.linear_model import Ridge,Lasso

# 参数越多、模型越简单、相同的alpha惩罚力度越大
reg_rid = Ridge(alpha=0.005)
reg_rid.fit(X, y)
print(mean_squared_error(reg_rid.predict(X), y))

# 观察惩罚效果
t = np.linspace(0, 1, 200)
plt.subplot(121)
plt.plot(x, y, 'ro', x, reg_rid.predict(X), '-', t, np.sqrt(t), 'r--')
plt.title('Ridge(alpha=0.005)')
plt.subplot(122)
plt.plot(x, y, 'ro', x, lr.predict(X), '-', t, np.sqrt(t), 'r--')
plt.title('LinearRegression')
plt.show()

reg_las = Lasso(alpha=0.001)
reg_las.fit(X, y)
mean_squared_error(reg_las.predict(X), y)
t = np.linspace(0, 1, 200)
plt.subplot(121)
plt.plot(x, y, 'ro', x, reg_las.predict(X), '-', t, np.sqrt(t), 'r--')
plt.title('Lasso(alpha=0.001)')
plt.subplot(122)
plt.plot(x, y, 'ro', x, lr.predict(X), '-', t, np.sqrt(t), 'r--')
plt.title('LinearRegression')

#Lasso的惩罚力度更强，并且迅速将一些参数清零，
# 而这些被清零的参数，则代表对应的参数在实际建模过程中并不重要，
# 从而达到特种重要性筛选的目的。

# l2正则化往往应用于缓解过拟合趋势，而l1正则化往往被用于特征筛选的场景中。


#当模型效果（往往是线性模型）不佳时，可以考虑通过特征衍生的方式来进行数据的“增强”；

#如果出现过拟合趋势，则首先可以考虑进行不重要特征的筛选，
# 过多的无关特征其实也会影响模型对于全局规律的判断，
# 当然此时可以考虑使用l1正则化配合线性方程进行特征重要性筛选，剔除不重要的特征，保留重要特征；

# 对于过拟合趋势的抑制，仅仅踢出不重要特征还是不够的，对于线性方程类的模型来说，
# l2正则化则是缓解过拟合非常好的方法，配合特征筛选，能够快速的缓解模型过拟合倾向；