"""
标题：python读取MNIST数据集；
时间：2022年8月21日；
说明：对MNIST手写数字数据文件转换为bmp图片文件格式。数据集下载地址为http://yann.lecun.com/exdb/mnist。
      相关格式转换见官网以及代码注释。
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle

# 设置文件路径
# 训练集文件
train_images_file = 'Datasets/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_file = 'Datasets/train-labels.idx1-ubyte'
# 测试集文件
test_images_file = 'Datasets/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_file = 'Datasets/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # print("bin_data:", bin_data)

    # 解析文件头信息，一次为魔法数量、图片数量、每张图片的行数和列数点阵
    offset = 0
    fmt_header = '>iiii'  # '>iiii'是说使用大端法读取4个unsigned int32，有几行就放几个i，前面用“>"
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print("魔法数量：{0}, 图片数量：{1}，单张图片大小：{2}*{3}".format(magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset:", offset)
    fmt_image = '>' + str(image_size) + 'B'  # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows * num_cols))  # 随机产生矩阵或队列,6万个(28*28)的矩阵
    print('images:\n', images)
    print('shape-images:\n', images.shape)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("已解析{0}张".format(i+1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        # print("images[{0}]:{1}".format(i, images[i]))
        offset += struct.calcsize(fmt_image)
    print("images.T:{0}".format(images.T))
    print("shape-iamges.T:{0}".format(images.T.shape))
    # print(images.T)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return:数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，一次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("魔数：{0},图片数量：{1}张".format(magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)  # 6000 * 1
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("已解析{0}张".format(i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]  # 无[0]返回结果为一个元祖：(3,0)，故取第一元素值
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n * row * col 维 np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象,n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def data_normalize_images(images_data, train_labels, test_images, test_labels):
    """
    数据标准化：把0-255数据标准化到[0.01,  0.99]范围内
    :param images_data: 训练图形数据
    :param train_labels: 图形标签
    :param test_images: 测试图形数据
    :param test_labels: 测试图形标签
    :return:
    """
    frac = 0.99/255
    print("images_data:\n", images_data)
    norm_train_imgs = np.asfarray(images_data * frac + 0.01)
    print("train_imgs:\n", norm_train_imgs)  # 训练数据标准化
    print("train_biaoqian:", train_labels)   # 训练数据标签
    print("train_biaoqian-type:", type(train_labels), ", shape:", train_labels.shape)  # 标签类型和大小

    norm_test_imgs = np.asfarray(test_images * frac + 0.01)  # 测试数据标准化
    print("norm_test_imgs:\n", norm_test_imgs)
    print("test_标签：\n", test_labels)
    print("test_biaoqian-type:", type(test_labels), ", shape:", test_labels.shape)  # 标签类型和大小

    # 把labels转变成one hot representation. 5:[0, 0., 0, 0, 0, 1, 0, 0, 0, 0]  2:[0, 0, 1, 0, 0, 0, 0 ,0 , 0, 0] [4., 5., ]
    train_labels_one_hot = np.identity(10)[train_labels.astype(int)]
    print("train_labels_one_hot:\n", train_labels_one_hot)
    test_labels_one_hot = np.identity(10)[test_labels.astype(int)]
    print('test_labels_one_hot:\n', test_labels_one_hot)

    # 把one hot表达式的labels转换
    train_labels_one_hot[train_labels_one_hot == 0] = 0.01
    train_labels_one_hot[train_labels_one_hot == 1] = 0.99
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99

    # 把每个矩阵放入到文件中
    with open('Datasets/pickled_mnist3.pkl', 'bw') as fh:
        data = (norm_train_imgs, norm_test_imgs, train_labels, test_labels, train_labels_one_hot, test_labels_one_hot)
        pickle.dump(data, fh)

    print("end!")


def run():
    train_images = load_train_images()  # (num_images, num_rows*num_cols)
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 数据标准化
    data_normalize_images(train_images, train_labels, test_images, test_labels)

    print("end!")


if __name__ == '__main__':
    run()
