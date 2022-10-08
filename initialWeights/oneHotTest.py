"""
标题：Numpy的One-hot形式的转换（eye，identity）
时间：2022年8月21日
"""

import numpy as np

# Numpy.eye()返回1个斜面全部为1，其他全部为0的2维的ndarray
eye = np.eye(4)
print(type(eye))
print(eye)
print(eye.dtype)

# 默认的数据类型为float64，参数dtype可以指定数据的类型；
# 参数M可以指定列数，参数k可以指定1开始的位置。
eye2 = np.eye(4, M=3, k=1, dtype=np.int8)
print(eye2)
print(eye2.dtype)

# numpy.identity()是一个可以返回单位矩阵（identity matrix）的函数。
i = np.identity(4)
print('i1:\n', i)
print('dtype:', i.dtype)

# 默认的数据类型为float64，参数dtype可以指定数据的类型。
# 无其他参数。
i2 = np.identity(4, dtype=np.uint8)
print('i2:\n', i2)
print(i2.dtype)

label = [3, 0, 8, 1, 9]
a_one_hot = np.identity(10)[label]  # 将label作为行数的索引放在后面的方括号中即可
print('a_one_hot:\n', a_one_hot)

a2 = [2., 2., 0., 1, 0]
a2 = np.array(a2).astype(int)
print("a-type:", type(a2))
a_one_hot2 = np.identity(3)[a2]
print(a2)
print(a_one_hot2)
















