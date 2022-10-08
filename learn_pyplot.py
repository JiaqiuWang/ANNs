# 1.导入必要的包。Matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
# 绘制一张空白图
fig, ax = plt.subplots()  # 创建图实例
x = np.linspace(0, 2, 100)  # 创建x的取值范围
y1 = x
ax.plot(x, y1, label='linear')  # 作y1=x图，并标记此线名为linear


# 如法炮制在同一张图上绘制多条曲线
y2 = x ** 2
ax.plot(x, y2, label='quadratic')  # 作y2=x^2图，并标记此线名为quadratic\
y3 = x ** 3
ax.plot(x, y3, label='cubic')  # 作y3=x^3图，并标记此线名为cubic

# 设置轴的名称和图名，并显示。
ax.set_xlabel('x label')  # 设置x轴的名称
ax.set_ylabel('y label')  # 设置y轴名称
ax.set_title('Simple Plot')  # 设置图名
ax.legend()  # 自动检测要在图例中显示的元素并显示

plt.show()  # 图形可视化
