"""
标题：卷积神经网络：采用Keras+TensorFlow识别手写体数字；
时间：2022年9月7日；
作者：王佳秋
"""

from numpy import mean
from numpy import std

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # one_hot函数
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D  # 解决图像识别的，Conv3D用于视频
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense  # 稠密连接的函数
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


# 1.加载数据：加载训练数据和测试数据
def load_dataset():
    """
    :return:
    """
    # (1)加载MNIST手写体数据
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    print('trainX shape:', trainX.shape)
    print('trainY shape:', trainY.shape)
    # (2)重整形状到一种单一的通道reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # (3)转换成one hot representation: one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # (4)打印前5行，看one hot representation是什么样
    for i in range(5):
        print('trainY[{0}]:{1}'.format(i, trainY[i]))
    # (5)返回读取后的数据集
    return trainX, trainY, testX, testY


# 2.数据标准化scale pixels
def prep_pixels(train, test):
    """
    :param train:
    :param test:
    :return:
    """
    # (1)把样本中的值，从整数转化成浮点数convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # (2)把0转成0.01,1转成0.99
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # (3)返回预处理后的数据集
    return train_norm, test_norm


# 3.配置学习模型
def define_model():
    """
    :return:
    """
    # (1)初始化序列为模型7
    model = Sequential()
    # (2)加入一个卷积层,包含8个filter过滤器，每个过滤器为3*3大小的,每个filter用'he_uniform'方法初始化值。
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # 每层输出结果数量的计算公式(formula for calculate the number of output for each layer)
    # output = (input - kernel + 2*padding)/Stride + 1 这里写的不对，参考知乎收藏里的说明。
    # 输出宽度=（输入图像的宽度-滤波器或内核的宽度+2*填充）/移动步幅 + 1 = (28-3+2*0)/1 + 1= 26
    # 输出高度=（输入图像的高度-滤波器或内核的高度+2*填充）/移动步幅 + 1 = (28-3+2*0)/1 + 1= 26
    # 整体输出结果的shape为8个(26, 26, 1)
    # (3)加入一个池化层
    model.add(MaxPooling2D((2, 2)))
    # output = (input - kernel + 2*padding)/Stride + 1 这里写的不对，参考知乎收藏里的说明。
    # 输出宽度=（输入图像的宽度-滤波器或内核的宽度+2*填充）/移动步幅 + 1 = (26-2+2*0)/2 + 1= 13
    # 输出高度=（输入图像的高度-滤波器或内核的高度+2*填充）/移动步幅 + 1 = (26-2+2*0)/2 + 1= 13
    # 整体输出结果的shape为8个(13, 13, 1)，拉平后8*(13*13)为输入向量的维度
    # (4)将池化层数据拉平
    model.add(Flatten())  # 注意：需要计算拉平后的输出
    # (5)拉平后送给全连接层
    model.add(Dense(120, activation='relu', kernel_initializer='he_uniform'))  # 隐藏层
    # (6)全连接后+一个输出层
    model.add(Dense(10, activation='softmax'))
    # (7)配置模型compile
    # 定义优化器为SGD随机梯度下降
    # 除了SGD以外，常用的优化器还有RMSprop, Adam, Adagrad, Adamax, Nadam, Ftrl
    # momentum 项能够在相关方向加速SGD
    # metrics是对模型有效性、性能performance的测量，classification分类问题与regression回归问题的测量方法不同
    # Keras对classification分类问题支持的测量包括：Binary Accuracy, Categorical Accuracy, Saprese Categorical Accuracy,
    # Top K, Sparese Top K
    opt = SGD(lr=0.01, momentum=0.9)  # 随机梯度下降优化
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print('模型基础信息：', model.summary())
    # (8)返回模型
    return model


# 4.评估模型：evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    """
    :param dataX:
    :param dataY:
    :param n_folds:
    :return:
    """
    # (1)初始化分数与历史为List类型
    scores, histories = list(), list()
    # (2)准备交叉验证
    kflod = KFold(n_folds, shuffle=True, random_state=1)
    # (3)数据放入到kfold中
    for train_ix, test_ix in kflod.split(dataX):  # train_ix，test_ix都是索引index值
        # (4)每一折用一个模型
        model = define_model()
        # (5)提取训练数据、标签与测试数据、标签: select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # (6)训练模型fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=60, validation_data=(testX, testY), verbose=0)
        # (7)增加几个打印语句，方便调式与程序理解
        print('打印训练模型历史信息：', history.history.keys())
        # (8)评估模型evaluate model
        loss, acc = model.evaluate(testX, testY, verbose=0)
        print('>%.3f' % (acc*100.0))
        # (8)把每一折fold的值存在scores中,history
        scores.append(acc)
        histories.append(history)
    print('scores:', scores)
    print('histories.len:', len(histories))
    # 返回数值
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

    plt.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean= %.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


# 0.运行测试工具来评估模型run the test harness for evaluating a model
def run_mymodel_test():
    # (1)加载数据
    trainX, trainY, testX, testY = load_dataset()
    # (2)数据预处理：准备数据集，像素数据预处理，转化成浮点数，并压缩到0-1之间
    trainX, testX = prep_pixels(trainX, testX)
    # (3)模型评估，其中先构造模型，再调用学习
    scores, histories = evaluate_model(trainX, trainY)
    # (4)打印学习曲线，看学习的过程、趋势变化
    summarize_diagnostics(histories)
    # (5)总结模型的性能performance
    summarize_performance(scores)


# 主程序人口
run_mymodel_test()
