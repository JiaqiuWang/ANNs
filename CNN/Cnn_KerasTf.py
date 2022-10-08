"""
标题：采用Keras+TensorFlow识别手写体数字；
时间：2022年9月6日；
作者：王佳秋
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # 稠密连接的函数
from tensorflow.keras import utils
import matplotlib.pyplot as plt

# 1.装载数据 load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# ndarray:(60000, 28, 28), (60000,), (10000, 28, 28), (10000,)
# 2.对于每一个图像，拉平28*28图像为一个784维度的向量
num_pixels = X_train.shape[1] * X_train.shape[2]  # 28 * 28
# 3.把X_train变成(60000, 784)形状，且每个元素变成单浮点数据类型
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# 4.数据标准化：对输入数据从[0,255]标准化到(0, 1)normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# 5.构造分类或label的one hot representation
y_train = utils.to_categorical(y_train)  # (60000, 10)
y_test = utils.to_categorical(y_test)  # (10000, 10)
# 6.获取分类数量
num_classes = y_test.shape[1]  # 10类

# 定义基线模型define baseline model
def baseline_model():
    # 1.创建模型create model
    # 顶一个一个顺序模型或者称序贯模型，与之对应的还有functional API模型
    model = Sequential()
    # 2.配置一个隐藏层，带一个输入层input_dim指定,kernel_initializer='normal'表示初始权重采用正态分布方法
    model.add(Dense(100, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # 3.配置一个输出层
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # 4.配置/组装模型compile model：指定损失函数、优化器、测量值（用于评估模型Performance）
    # 可选的优化器有SGD,RMSprop,Adam,Adadelta,Adagrad,Adamax,Nadam,Ftrl
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 5.返回模型
    return model


# 7.构建模型
model = baseline_model()
# 8.开始学习，10遍epochs，以每个批量100个样本，每运行完100个样本，更新一次权重值
# 跑一百个以后更新权重值：一个样本运行后记住delta权重，第二个记住delta权重，跑到100个后把delta值都加起来后用整体更新W
# verbose=2表示打印中间过程信息
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=2)
# 9.最后评估模型final evaluation of the model
# 返回模型的评估结果，得分，其中返回的第一个值是Loss，第二个视是在compile方法中的metrics会有所不同。
# 若只指定了Accuracy，只返回准确率  verbose=0不打印中间细节信息
scores = model.evaluate(X_test, y_test, verbose=0)
print('Scores:', scores)
print('Baseline Error: %.3f%%' % (100-scores[1]*100))  # 损失保留3位小数
print("end!")


