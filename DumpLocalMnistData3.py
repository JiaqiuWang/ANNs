"""
标题：python读取MNIST数据集；
时间：2022年8月21日；
说明：对MNIST手写数字数据文件转换为bmp图片文件格式。数据集下载地址为http://yann.lecun.com/exdb/mnist。
      相关格式转换见官网以及代码注释。
"""

import numpy as np
import struct
import matplotlib.pyplot as plt

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


def run():
    train_images = load_train_images()  # (num_images, num_rows*num_cols)
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print("训练图形：\n", train_images[i])

    # for i in range(10):
    #     print(test_labels[i])

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(32):
        images = np.reshape(train_images[i], [28, 28])
        ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(train_labels[i]))
    plt.show()

if __name__ == '__main__':
    run()



