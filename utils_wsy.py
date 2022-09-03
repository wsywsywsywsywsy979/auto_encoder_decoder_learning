"""
    author:wsy
    date:2022-6-26
    放置一些常用函数
"""
import numpy as np
from matplotlib import pyplot as plt
import torch
def load_data():
    """
    读取数据
    """
    path = r'已看论文\mycode\mnist.npz'
    f=np.load(path)
    # X矩阵是28x28 numpy数组，而y只是一个整数。
    x_train, y_train = f['x_train'], f['y_train']  # (60000, 28, 28), (60000,)
    x_test, y_test = f['x_test'], f['y_test']  # (60000, 28, 28), (10000,)
    f.close()
    x_train = x_train.astype('float32') / 255.  # 归一化
    x_test = x_test.astype('float32') / 255.  # 归一化
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  # (60000, 784)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  # (60000, 784)
    return x_train,x_test,y_train,y_test

def draw_result(x,y):
    ax=plt.subplot(1,3,1) # 参数：行，列，索引
    plt.imshow(x.reshape(28,28)) # 原数据
    plt.gray() # 展示灰度图
    ax=plt.subplot(1,3,2)
    plt.imshow(y.data.reshape(28,28))
    plt.gray()
    plt.show()

# 以下的采样函数是vae和cvae通用的，所以抽取出来
def sampling(args):
    """
    从拟合的隐分布中采样，这个就是重参数化！！
    """
    mu,log_var=args
    std = torch.exp(0.5 * log_var) # diff
    eps = torch.randn_like(std) 
    # 返回一个和输入大小相同的张量：由均值为0、方差为1的标准正态分布填充,因为噪声是要和方差进行元素级乘积的
    return mu + eps * std

def vae_loss(x,de_mean,z_log_sigma,z_mean):
    """
    定义损失函数
    """
    BCE = torch.nn.functional.binary_cross_entropy(
        de_mean.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 +z_log_sigma - z_mean.pow(2) - z_log_sigma.exp())
    return (BCE + KLD) / x.size(0)