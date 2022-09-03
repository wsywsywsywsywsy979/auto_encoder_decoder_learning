"""
    author:wsy
    date:2022-6-21
"""
import  torch
# import torch.nn as nn # 该模块包含了常用层和损失函数
import numpy as np
# import torch.optim as optim # 优化器所需模块
from torch import nn,optim
from torch.autograd import Variable # 受梯度影响的变量
import torch.nn.functional as F
from os import sys, path
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(path.join(PARENT_DIR,"mycode"))
import utils_wsy

class my_vanilla_autoencoder(nn.Module):
    # 香草编码器
    def __init__(self) :
        super(my_vanilla_autoencoder, self).__init__() # 创建基类的一个实例
        encoding_dim = 32  # 压缩维度
        # 定义网络结构 （3层网络），如此定义就是有3层网络了
        self.encoded=nn.Linear(784,encoding_dim) # 输入层
        self.decoded=nn.Linear(encoding_dim,784) # 输出层
    
    def forward(self,x):
        return F.sigmoid(self.decoded(F.relu(self.encoded(x))))

if __name__=="__main__":
    # 基本自编码器：香草编码器
    va=my_vanilla_autoencoder()
    print(va) # 打印网络结构
    x_train,x_test,_,_=utils_wsy.load_data() # x_train:shape(60000,784)
    opt=optim.Adam(va.parameters(),lr=0.01) # 需要将所有参数传给优化器
    lossf=nn.MSELoss()
    # 训练：
    epochs=10 # 训练10个epoch
    batch_size=512 # 设置批处理大小为512
    index=0
    size=x_train.shape[0] # 这里可能需要用shape[0]
    for epoch in range(epochs):
    #     # for batch_idx,(data,target) in enumerate(train_loader):
        index=0
        while index+batch_size < size:
            data=np.array(x_train)[index:index+batch_size,:]
            index+=batch_size
            data=Variable(torch.tensor(data)) #参数需要是tensor
            # data=Variable(np.ndarray(data)) # 将数据变为pytorch张量,只有array类型可以直接这么转
            opt.zero_grad() # 让模型中所有梯度归零重置,为下一次反向传播做准备
            output=va(data) # 将数据传入模型，实际是调用forward方法
            loss=lossf(output,data) # 计算损失
            loss.backward() # 损失回传
            opt.step() # 逐步优化
            if index%(batch_size*100)==0: # 每100个批次输出一次
                a=loss.data.numpy().tolist()
                print("epoch"+str(epoch)+":loss----"+str(a)+",index----"+str(index)) # 注意此处输出loss需要访问data属性
    
    # 测试模型：
    test_loss=0
    correct=0
    index=0
    for i in range(10):
        data=x_test[i]
        data=Variable(torch.tensor(data)) 
        output=va(data)
        utils_wsy.draw_result(data,output)
        # 损失累加求和
        test_loss+=lossf(output,data).data.numpy().tolist()
    test_loss/=10
    print("average test loss:"+str(test_loss))