"""
    正则编码器中的降噪编码器
    author:wsy
    date:2022-6-22
"""
import torch
import torch.nn as nn
from torch import optim
from os import sys, path
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(path.join(PARENT_DIR,"mycode"))
import utils_wsy
class noise_reduce_autoencoder(nn.Module):
    def __init__(self) -> None:
        super(noise_reduce_autoencoder,self).__init__()
        # 定义网络结构
        ## 编码器 
        self.en_c1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.en_c2=nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        ## 解码器
        self.de_c1=nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2))
        )
        self.de_c2=nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2))
        )
        self.c3=nn.Sequential(
            nn.Conv2d(32,1,3,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.en_c1(x) # (512, 1, 28, 28)->(512, 32, 14, 14)
        x=self.en_c2(x) #                 ->(512, 32, 7, 7)
        x=self.de_c1(x) #                 ->(512, 32, 14, 14)
        x=self.de_c2(x) #                 ->(512, 32, 28, 28)
        x=self.c3(x)    #                 ->(512, 1, 28, 28)
        return x

if __name__=="__main__":
    # 创建噪声数据
    noise_factor=0.5 # 噪声因子
    x_train,x_test,_,_=utils_wsy.load_data()
    #---------因为涉及卷积所以此处需要变为二维
    x_train=np.reshape(x_train,(len(x_train),28,28,1))
    x_test=np.reshape(x_test,(len(x_test),28,28,1))
    x_train_noise=x_train+noise_factor*np.random.normal(size=x_train.shape)
    x_test_noise=x_test+noise_factor*np.random.normal(size=x_test.shape)
    x_train_noise=np.clip(x_train_noise,0.,1.) # 限制最大和最小值
    x_test_noise=np.clip(x_test_noise,0.,1.)
    # 训练
    nr=noise_reduce_autoencoder()
    opt=optim.Adadelta(nr.parameters())
    # lossf=nn.MSELoss() # 奇怪每次使用交叉熵函数都是一开始的损失就是0
    # lossf=F.binary_cross_entropy() # 因为使用均方差损失函数得到的结果太差了
    epochs=10 # 训练10个epoch
    batch_size=512 # 设置批处理大小为512
    index=0
    size=x_train.shape[0] # 这里可能需要用shape[0]
    for epoch in range(epochs):
        index=0
        while index+batch_size < size:
            data=np.array(x_train_noise)[index:index+batch_size,:] # ！！！ 此处输入改为了噪声
            data=data.swapaxes(1,3)
            data=Variable(torch.tensor(data)) #参数需要是tensor
            opt.zero_grad() # 让模型中所有梯度归零重置,为下一次反向传播做准备
            # 需要调整维度
            output=nr(data.to(torch.float32)) # 将数据传入模型，实际是调用forward方法
            target=np.array(x_train)[index:index+batch_size,:]  # 目标是未加噪声的数据
            target=target.swapaxes(1,3)
            index+=batch_size
            target=Variable(torch.tensor(target))
            loss=F.binary_cross_entropy(output,target) # 二元分类交叉熵函数
            loss.backward() # 损失回传
            opt.step() # 逐步优化
            if index%(batch_size*100)==0: # 每100个批次输出一次
                a=loss.data.numpy().tolist()
                print("epoch:"+str(epoch)+":loss----"+str(a)+",index----"+str(index)) # 注意此处输出loss需要访问data属性
    # 测试模型：
    test_loss=0
    correct=0
    index=0
    for i in range(10):
        data=x_test_noise[i] # 使用测试集中加了噪声的数据进行测试
        data=data.swapaxes(0,2) # 交换维度
        data=data[np.newaxis,:] # 增加一个维度便于匹配网络（因为网络训练是批量训练的，多一个维度）
        data=Variable(torch.tensor(data)) 
        output=nr(data.to(torch.float32))
        target=x_test[i] # 使用未加噪声的作为目标
        target=target.swapaxes(0,2)
        target=target[np.newaxis,:]
        target=Variable(torch.tensor(target))
        utils_wsy.draw_result(target,output)
        # 损失累加求和
        test_loss+=F.binary_cross_entropy(output,target)
    test_loss/=10
    print("average test loss:"+str(test_loss))