import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# Rand Data / Test
np.random.seed(0)
x = np.linspace(0, 10, 50)
y = 2.5 * x + 1.2 + np.random.normal(0, 1, size=x.shape)

def model_function(x, a, b):
    # Function models
    return a * x + b

params, covariance = curve_fit(model_function, x, y)

print(f"拟合结果: y = {params[0]:.2f}x + {params[1]:.2f}")

# PIC
plt.scatter(x, y, label='原始数据')
plt.plot(x, model_function(x, *params), 'r', label='拟合直线')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('拟合')
plt.grid(True)
plt.show()