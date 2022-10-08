
import numpy as np
import matplotlib.pyplot as plt
import struct


def load_images(file_name):
    """
    在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。
    file object = open(file_name [, access_mode][, buffering])
    file_name是包含您要访问的文件名的字符串值。
    access_mode指定该文件已被打开，即读，写，追加等方式。
    0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
    这里rb表示只能以二进制读取的方式打开一个文件
    """
    binfile = open(file_name, 'rb')
    #  从一个打开的文件读取数据
    buffers = binfile.read()
    #  读取image文件前4个整型数字
    magic, num, rows, cols, other = struct.unpack_from('>IIIII', buffers, 0)
    print(magic, num, rows, cols, other)
    #   整个images数据大小为60000*28*28
    bits = num * rows * cols
    #  读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    #  关闭文件
    binfile.close()
    #  转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    print('images:\n', images)
    return images


def load_labels(file_name):
    # 打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic,num = struct.unpack_from('>II', buffers, 0)
    print("magic_num:", magic, " num:", num)
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels

filename_train_images = 'Datasets/train-images.idx3-ubyte'
filename_train_labels = 'Datasets/train-labels.idx1-ubyte'
filename_test_images = 'Datasets/t10k-images.idx3-ubyte'
filename_test_labels = 'Datasets/t10k-labels.idx1-ubyte'
train_images=load_images(filename_train_images)
print("train_images-shape:", train_images.shape)
train_labels=load_labels(filename_train_labels)
test_images=load_images(filename_test_images)
test_labels=load_labels(filename_test_labels)

fig=plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(32):
    images = np.reshape(train_images[i], [28,28])
    ax=fig.add_subplot(6,6,i+1,xticks=[],yticks=[])
    ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,7,str(train_labels[i]))
plt.show()

print("end!")
