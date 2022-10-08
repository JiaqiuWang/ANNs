"""
标题：Ann识别手写体；
时间：2022年8月3日；
说明：
"""

import numpy as np
import pickle
from scipy.stats import truncnorm


# numpy.vectorize takes a function f:a->b and turns it into g:a[]->b[]
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    pass


activation_function = sigmoid

# 引入scipy的统计包，目的是引入truncnorm函数
# truncnorm无法指定上下界，稍微改造让它体现上下界


# 生成有边界的均值为0，标准差为1的正态分布数据模式；
def truncated_normal(mean=0, sd=1, low=0.0, upp=10.0):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# 定义神经网络类ANN
class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes  # 输入节点数量
        self.no_of_out_nodes = no_of_out_nodes  # 输出节点数量
        self.no_of_hidden_nodes = no_of_hidden_nodes  # 隐藏节点数量
        self.learning_rate = learning_rate  # 学习速率
        self.create_weight_matrices()  # 初始化时，调用NN网络的权重矩阵

    def create_weight_matrices(self):
        """
        Initialize the weight matrices：初始化权重矩阵
        :return:
        """
        # 定义上下界：（输入节点数+偏移量节点数）开平方根分之一
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)  # 定义均值0，标准差1，上下界分布
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))  # 按照分布情况，产生服从指定分布的随机数
        print('wih-shape:', self.wih.shape, ", \n", self.wih)

        # 隐藏层与输出层之间的权重
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
        print('who-shape:', self.who.shape, ", \n", self.who)

    # 定义训练函数；
    def train(self, input_vector, target_vector):
        """
        输入一次参数，执行一次函数；循环函数在其它位置
        :param input_vector: (60000, 784)
        :param target_vector: (60000, 10)
        :return:
        """
        # 做准备工作，让input_vector符合模型的dot product运算的输入
        # ndmin ： int，指定结果数组应具有的最小维数。根据需要，将根据需要预先设置形状。
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # between input & hidden
        # 做输入层与隐藏层的点积运算，结果放入激活函数得到（隐藏层）输出
        output_vector1 = np.dot(self.wih, input_vector)  # (100, 784) * (784, 60000) = (100, 60000)
        output_hidden = activation_function(output_vector1)  # (100, 60000)

        # between hidden & output
        # 做隐藏层与输出层的点积运算，结果放入激活函数得到（输出层）输出
        output_vector2 = np.dot(self.who, output_hidden)  # (10, 100) * (100, 60000) = (10, 60000)
        output_network = activation_function(output_vector2)  # (10, 60000)

        # 目标值减去输出值作为损失函数
        output_errors = target_vector - output_network  # (60000, 10).T - (10, 60000) = (10, 60000)

        # update the weights，实施back propagating,损失函数做梯度下降，gradient descent，结果更新权重
        tmp = output_errors * output_network * (1.0 - output_network)  # (10,60000)*(10,60000)*(1-(10,60000))
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)  # 0.1 * (10, 60000) * (100, 60000).T = (10, 100)
        # 得到心得隐藏层与输出层之间的权重矩阵
        self.who += tmp  # 为何用+号，参考公式：第80行与公式相反了，减去一个减号变成了加号  new_who=(10, 100)+(10, 100)

        # calculate hidden errors，通过新的who权重矩阵和输出损失函数计算隐藏层的损失
        hidden_errors = np.dot(self.who.T, output_errors)  # (10,100).T * (10, 60000) = (100, 60000)

        # 更新隐藏层的权重；得到输入层与隐藏层的权证矩阵wih
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)  # (100, 60000)*(100, 60000)*(1-(100, 60000))
        # (4, 1) * (4, 1) * (4, 1)
        tmp = self.learning_rate * np.dot(tmp, input_vector.T)  # 0.1 * (100, 60000) * (784, 60000).T=(100, 784)
        # C * dot((4, 1), (3, 1).T) = (4, 3)
        self.wih += tmp  # new_wih = old_new_wit + tmp = (100, 784) + (100, 784) = (100, 784)

    def run(self, input_vector):
        """
        测试函数或预测函数，检查分类的真确性，主要看学习到的权重有效性
        :param input_vector: 模型输入的矩阵,一个向量test_imgs[i].shape = (784,)->input_vector
        :return: 模型输出的分类结果为一个10分类向量
        """
        # print("input_vector-shape:", input_vector.shape, ", ndmin=2转换后：", np.array(input_vector, ndmin=2).shape)
        # input_vector-shape: (784,) , ndmin=2转换后： (1, 784)
        input_vector = np.array(input_vector, ndmin=2).T  # (1, 784).T
        output_vector = np.dot(self.wih, input_vector)  # (100, 784)* (784, 1) = (100, 1)
        output_vector = activation_function(output_vector)  # (100, 1)
        output_vector = np.dot(self.who, output_vector)  # (10, 100) * (100, 1) = (10, 1)
        output_vector = activation_function(output_vector)  # (10, 1)
        # print("inside run output_vector:", output_vector)
        return output_vector  # 返回(10, 1)矩阵

    def confusion_matrix(self, data_array, labels):
        """
        通过调用run函数，获得模型学习结果的有效性
        :param data_array: 训练数据集(60000, 784)
        :param labels: (60000, )
        :return:
        """
        print("data_array-shape:", data_array.shape, ", labels-shape:", labels.shape)
        cm = np.zeros((10, 10), int)  # 初始化混淆矩阵为10行*10列的0矩阵，datatype为int
        for i in range(len(data_array)):
            res = self.run(data_array[i])  # 将数据中每一行放入到预测行数中返回预测结果（10,1）矩阵
            # print("in cm loop res is:", res)
            res_max = res.argmax()
            # print("res_max:", res_max)
            # print('labels[i]:', labels[i], ', type:', type(labels[i]))
            target = labels[i]  # 一个数字：0.-9.

            # cm[res_max, int(target)] += 1  # cm[5, 5] = 1; cm[5, 4] = 1
            cm[int(target), res_max] += 1  # 行为真实数据、列为预测数据；cm[5, 5] = 1; cm[5, 4] = 1
            # print('in cm loop CM is:\n', cm)
            # if res_max != int(target):
            #     break
        return cm

    def precision(self, label, confusion_matrix):
        """
        计算精准率，查准率
        :param label:
        :param confusion_matrix:
        :return:
        """
        col = confusion_matrix[:, label]  # [行, 列]，所以行，第label列
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        """
        计算召回率，查全率
        :param label:
        :param confusion_matrix:
        :return:
        """
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):  # (60000, 784)， (60000,)
        """
        计算分对与分错的数量
        :param data:
        :param labels:
        :return:
        """
        corrects, wrongs = 0, 0
        for i in range(len(data)):  # 60000
            res = self.run(data[i])  # data[i].shape=(784,), 返回(10, 1)输出结果矩阵
            res_max = res.argmax()  # 选出结果矩阵中最大概率的索引
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


def main_run():
    np.set_printoptions(suppress=True)  # 不采用np默认的科学计数法表示数据
    # 1.装载数据
    with open('Datasets\pickled_mnist.pkl', 'br') as fh:
        data = pickle.load(fh)
    # 2.按顺序读取数据
    train_imgs = data[0]  # norm数据标准化后的训练集
    test_imgs = data[1]
    train_labels = data[2]
    test_labels = data[3]
    train_labels_one_hot = data[4]
    test_labels_one_hot = data[5]

    print("train_imgs-shape:", train_imgs.shape, ", test_imgs-shape:", test_imgs.shape, ", train_labels-shape:",
          train_labels.shape, ", test_labels-shape:", test_labels.shape, ", train_labels_one_hot-shape:",
          train_labels_one_hot.shape, ", test_labels_one_hot-shape:", test_labels_one_hot.shape)

    image_size = 28
    no_of_different_labels = 10
    image_pixels = image_size * image_size

    # initialize an ANN，初始化一个ANN
    ANN = NeuralNetwork(no_of_in_nodes=image_pixels, no_of_out_nodes=10, no_of_hidden_nodes=100, learning_rate=0.1)

    # 学习一边
    print("len(train_imgs) before learning:", len(train_imgs))

    for i in range(len(train_imgs)):  # 所有训练样本训练一次
        ANN.train(train_imgs[i], train_labels_one_hot[i])
    # print("i is after learning:", i)

    for i in range(20):  # 用模型预测
        res = ANN.run(test_imgs[i])  # test_imgs[i].shape = (784,)
        # print('test_imgs[i]-shape:', test_imgs[i].shape)  # (784,)
        print("test_labels[{0}], predict_result:{1}, max_probability_class:{2}, "
              "Actual Value:{3}, 最大值索引:{4}".format(i, res, max(res), test_labels[i], np.argmax(res)))

    corrects, wrongs = ANN.evaluate(train_imgs, train_labels)  # (60000, 784)， (60000,)
    print("Accuracy train:", corrects / (corrects + wrongs))
    corrects, wrongs = ANN.evaluate(test_imgs, test_labels)  # (10000, 784)，(10000,)
    print("Accuracy test:", corrects / (corrects + wrongs))

    cm = ANN.confusion_matrix(train_imgs, train_labels)  # 输入训练数据(60000, 784)， (60000,)
    print("cm:\n", cm)

    for i in range(10):
        print('Digit:', i, ', Precision:', ANN.precision(i, cm), ', Recall:', ANN.recall(i, cm))

    print("End!")


if __name__ == '__main__':  # 主函数
    main_run()
