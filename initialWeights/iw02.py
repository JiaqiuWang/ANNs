import matplotlib.pyplot as plt
from scipy.stats import truncnorm


# mean:均值；sd：标准差
def truncated_normal(mean=0, sd=1.0, low=0.0, upp=10.0):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


X = truncated_normal(mean=0, sd=0.4, low=-0.5, upp=0.5)
s = X.rvs(size=10000)

plt.hist(s)
plt.show()



"""
通用方法
连续随机变量的主要公共方法如下：

rvs：随机变量（就是从这个分布中抽一些样本）
pdf：概率密度函数。
cdf：累计分布函数
sf：残存函数（1-CDF）
ppf：分位点函数（CDF的逆）
isf：逆残存函数（sf的逆）
stats：返回均值，方差，（费舍尔）偏态，（费舍尔）峰度。
moment：分布的非中心矩。
让我们使用一个标准正态(normal)随机变量(RV)作为例子。
"""