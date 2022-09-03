"""
    正则编码器中的稀疏编码器
    author:wsy
    date:2022-6-22
"""
import torch
from torch import optim
import numpy as np
from torch.autograd import Variable
from os import sys, path
import torch.nn.functional as F
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(path.join(PARENT_DIR,"mycode"))
from vanilla_autoencoder import my_vanilla_autoencoder
import utils_wsy

if __name__=="__main__":
    # 基本自编码器：香草编码器
    va=my_vanilla_autoencoder()
    print(va) # 打印网络结构
    x_train,x_test=va.load_data() # x_train:shape(60000,784)
    opt=optim.Adam(va.parameters(),lr=0.01) # 需要将所有参数传给优化器
    # 训练：
    epochs=10 # 训练10个epoch
    batch_size=512 # 设置批处理大小为512
    index=0
    size=x_train.shape[0] # 这里可能需要用shape[0]
    for epoch in range(epochs):
        index=0
        while index+batch_size < size:
            data=np.array(x_train)[index:index+batch_size,:]
            index+=batch_size
            data=Variable(torch.tensor(data)) #参数需要是tensor
            opt.zero_grad() # 让模型中所有梯度归零重置,为下一次反向传播做准备
            output=va(data) # 将数据传入模型，实际是调用forward方法
            loss=F.binary_cross_entropy(output,data) 
            #-----------增加L1正则化（模型权重的绝对值之和）----------------
            re_loss=0
            for para in va.parameters():
                re_loss+=torch.sum(torch.abs(para))
            loss+=0.01*re_loss
            #--------------------------------------------------------------
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
        data=x_test[i]
        data=Variable(torch.tensor(data)) 
        output=va(data)
        utils_wsy.draw_result(data,output)
        # 损失累加求和
        # test_loss+=lossf(output,data).data.numpy().tolist()
        test_loss+=F.binary_cross_entropy(output,data)
    test_loss/=10
    print("average test loss:"+str(test_loss))
