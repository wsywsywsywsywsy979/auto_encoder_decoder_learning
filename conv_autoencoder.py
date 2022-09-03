"""
卷积自编码器
    author:wsy
    date:2022-6-22
"""
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from os import sys, path
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(path.join(PARENT_DIR,"mycode"))
import utils_wsy

class conv_autoencoder(nn.Module):
    def __init__(self) -> None:
        super(conv_autoencoder, self).__init__()  # 继承父类的函数
        # 定义网络结构：
        ## 编码器:
        self.en_c1=nn.Sequential(
            nn.Conv2d(1,16,3,1,1), # 参数：输入图片的通道数（C），输出图片的通道，卷积核大小，stride,padding
            nn.ReLU(),
            nn.MaxPool2d(2) # 注意此处不要填充，并且上方padding为kernel_size/2是对的
        )
        self.en_c2=nn.Sequential(
            nn.Conv2d(16,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.en_c3=nn.Sequential(
            nn.Conv2d(8,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,padding=1)
        )
        ## 解码器 :
        self.de_c1=nn.Sequential(
            nn.Conv2d(8,8,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)) # 上采样,注意第一个参数是输出大小，此处需要使用关键字来给参数赋值
        )
        self.de_c2=nn.Sequential( # 对于这一层如何变化的还是有些不太对
            nn.Conv2d(8,8,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)) # 上采样 size和scale_factor只能指定一个 
        )
        self.de_c3=nn.Sequential( # 注意这一层不是same,即padding=0
            nn.Conv2d(8,16,3,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)) # 上采样 此处的参数应该是对应原始x的shape从右向左乘上倍数
        )
        self.de_c4=nn.Sequential(
            nn.Conv2d(16,1,3,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        """
        构建计算图
        """
        x=self.en_c1(x) # (512,1,28,28)->(512, 16, 14, 14)
        x=self.en_c2(x) #               ->(512, 8, 7, 7)
        x=self.en_c3(x) #               ->(512, 8, 4, 4)
        x=self.de_c1(x) #               ->(512, 8, 8, 8)
        x=self.de_c2(x) #               ->(512, 8, 16, 16)
        x=self.de_c3(x) #               ->(512, 16, 28, 28)
        x=self.de_c4(x) #               ->(512, 1, 28, 28)
        return x

if __name__=="__main__":
    cv=conv_autoencoder()
    print(cv)
    # 加载数据
    x_train,x_test,_,_=utils_wsy.load_data() # 直接复用加载数据
    x_train=np.reshape(x_train,(len(x_train),28,28,1))
    x_test=np.reshape(x_test,(len(x_test),28,28,1))
    # 训练：
    epochs=10
    batch_size=512 # 设置批处理大小为512
    index=0
    size=x_train.shape[0] # 这里可能需要用shape[0]
    opt=optim.Adadelta(cv.parameters())
    # lossf=nn.CrossEntropyLoss() # 使用交叉熵损失，使用这个函数导致一开始得到的损失就是0
    for epoch in range(epochs):
        index=0 # ！！！！
        while index+batch_size < size:
            data=np.array(x_train)[index:index+batch_size,:]
            index+=batch_size
            data=data.swapaxes(1,3) # array交换维度这个也是一个不改变源对象，然后返回新的对象的函数
            data=Variable(torch.tensor(data)) #参数需要是tensor
            opt.zero_grad() # 让模型中所有梯度归零重置,为下一次反向传播做准备
            output=cv(data) # 将数据传入模型，实际是调用forward方法
            loss=F.binary_cross_entropy(output,data) 
            loss.backward() # 损失回传
            opt.step() # 逐步优化
            if index%(batch_size*100)==0: # 每100个批次输出一次
                a=loss.data.numpy().tolist()
                print("epoch"+str(epoch)+":loss----"+str(a)+",index----"+str(index))
    # 测试：
    test_loss=0
    correct=0
    index=0
    for i in range(10):
        data=x_test[i]
        data=data.swapaxes(0,2) # 交换维度
        data=data[np.newaxis,:] # 增加一个维度便于匹配网络（因为网络训练是批量训练的，多一个维度）
        data=Variable(torch.tensor(data)) 
        output=cv(data)
        utils_wsy.draw_result(data,output)
        # 损失累加求和
        test_loss+=F.binary_cross_entropy(output,data)
    test_loss/=10
    print("average test loss:"+str(test_loss))