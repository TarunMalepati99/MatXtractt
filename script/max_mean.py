import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 假设数据
r1 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
r2 = np.array([2, 3, 4, 6, 9, 15])

# 可视化
plt.scatter(r1, r2)
plt.xlabel('r1')
plt.ylabel('r2')
plt.show()

# 相关性
correlation = np.corrcoef(r1, r2)[0, 1]
print(f"Correlation: {correlation}")

# 回归（这里用幂律回归作为例子）
def power_law(x, a, b):
    return a * np.power(x, b)

popt, _ = stats.curve_fit(power_law, r1, r2)

# 预测
r1_pred = 0.8
r2_pred = power_law(r1_pred, *popt)
print(f"Predicted r2 when r1=80%: {r2_pred}")

# 绘制拟合曲线
x_fit = np.linspace(0.2, 0.8, 100)
y_fit = power_law(x_fit, *popt)
plt.scatter(r1, r2)
plt.plot(x_fit, y_fit, 'r-')
plt.savefig('fig.pdf')
plt.show()