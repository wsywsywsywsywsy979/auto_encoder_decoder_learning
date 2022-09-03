"""
    author:wsy
    date:2022-6-26
"""
import torch
from torch import nn
from os import sys, path
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
# 获取所要引入的包的父目录，每一个path.dirname都是向上获取一次父目录
sys.path.append(path.join(PARENT_DIR,"mycode")) 
import utils_wsy
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
#------------------------定义全局变量----------------
# 定义超参数：
dim1=512 
latent_space=2 # 此处定义的隐变量维度也是2
batch_size=256
epochs=50
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
#---------------------------------------------------
class cvae_encoder(nn.Module):
    """
    cvae编码器
    """
    def __init__(self) -> None:
        super(cvae_encoder,self).__init__()
        # 定义网络结构：
        self.l1=nn.Sequential(
            nn.Linear(794,dim1), # 784->794 因为输入是x和one-hot y的拼接
            nn.ReLU()
        )
        self.mean=nn.Sequential(
            nn.Linear(dim1,latent_space),
            nn.ReLU() # 未找到pytorch中的线性激活函数，此处先使用relu
        )
        self.std=nn.Sequential(
            nn.Linear(dim1,latent_space),
            nn.ReLU() 
        )
    
    def forward(self,x,y): # x :(256, 784) y: (256,10)
        inputs=torch.cat([x,y],dim=1) # 和vae不同的地方   注意dim对应shape从右向左数（从0开始）
        x=self.l1(inputs) # (256, 512)   # 拼接之后编码器的第一层的全连接的输入是784，但拼接之后input维度变成了794
        z_mean=self.mean(x) # (256, 2)
        z_std=self.std(x)
        return z_mean,z_std

class cvae_decoder(nn.Module):
    """
    cvae解码器
    """
    def __init__(self) -> None:
        super(cvae_decoder,self).__init__()
        # 定义网络结构
        self.l1=nn.Sequential(
            nn.Linear(latent_space+10,dim1), # 因为此处输入是z和y_one_hot的拼接
            nn.ReLU()
        )
        self.out=nn.Sequential(
            nn.Linear(dim1,784),
            nn.Sigmoid()
        )
    
    def forward(self,zc):
        zc=self.l1(zc)
        out= self.out(zc)
        return out

class cvae_autoencoder(nn.Module):
    """
    CVAE自动编码器
    """
    def __init__(self) -> None:
        super(cvae_autoencoder,self).__init__()
        # 定义网络结构
        self.encoder=cvae_encoder()
        self.decoder=cvae_decoder()

    def forward(self,x,y):
        z_mean,z_std=self.encoder(x,y)
        z=utils_wsy.sampling([z_mean,z_std]) # z:(256, 2)
        # 将 隐变量和label连接：
        zc=torch.cat([z,y],dim=1) # z: (256, 2) y:(256, 10)  cat 对于形状不是完全相同的数组进行连接，需要指定维度相同的dim
        out=self.decoder(zc)
        return z_mean,z_std,out

def train(x_train,y_train):
    """
    训练
    """
    #-----------after first--------------
    cvae=torch.load("cvae.pt")
    #-----------------------------------
    #----------first--------------------
    # cvae=cvae_autoencoder().to(device)
    #-------------------------------------
    opt=torch.optim.Adam(cvae.parameters(),lr=0.001)
    size=x_train.shape[0]
    for epoch in range(epochs):
        index=0
        while index+batch_size < size:
            data=np.array(x_train)[index:index+batch_size,:]
            label=np.array(y_train)[index:index+batch_size,:]
            index+=batch_size
            data=Variable(torch.tensor(data)).to(device)
            label=Variable(torch.tensor(label)).to(device)
            opt.zero_grad()
            z_mean,z_log_sigma,de_mean=cvae(data,label)
            loss=utils_wsy.vae_loss(data,de_mean,z_log_sigma,z_mean)
            loss.backward()
            opt.step()
        a=loss.item()
        print("epoch:"+str(epoch)+":train_loss----"+str(a)+",index----"+str(index)) 
        print("##########################################")
    torch.save(cvae,"cvae.pt") 
    state={"z_mean":z_mean,"z_log_sigma":z_log_sigma}
    torch.save(state,"cvae_mean_sigma.pt")

def test(x_test,y_test):
    """
    计算测试集的损失
    """
    cvae=torch.load(r"自己训练的模型\cvae模型\cvae.pt")            
    index=0
    size=x_test.shape[0]
    loss=0
    with torch.no_grad():
        while index+batch_size < size:
            # 加载输入数据
            data=np.array(x_test)[index:index+batch_size,:]
            data=Variable(torch.tensor(data)).to(device)
            label=np.array(y_test)[index:index+batch_size,:]
            label=Variable(torch.tensor(label)).to(device)
            index+=batch_size
            # 前向传播：
            z_mean,z_log_sigma,de_mean=cvae(data,label)            
            # 计算损失并回传
            loss+=utils_wsy.vae_loss(data,de_mean,z_log_sigma,z_mean)
        print("test_loss:",loss.item())

def show(x,y,y_one_hot):
    """
    使用编码器生成潜在空间表示
    """
    cvae=torch.load(r"自己训练的模型\cvae模型\cvae.pt")
    encoder=cvae.encoder
    with torch.no_grad():
        x=torch.FloatTensor(x).to(device)
        y_one_hot=y_one_hot.to(device)
        z_mean,z_log_sigma=encoder(x,y_one_hot)
        plt.figure(figsize=(6, 6))
        z_mean=z_mean.detach().cpu().numpy()
        z_log_sigma=z_log_sigma.detach().cpu().numpy()
        plt.scatter(z_mean[:,0], z_log_sigma[:,0], c=y) # 此处参数需要是array类型，所以需要先转回来，注意三个参数的数量需要是一致的！
        plt.colorbar()
        plt.show()

def construct_numvec(n,z=None):
    """
    构建向量：该向量包含从潜在空间创建数字所需的所有内容。
    """
    # 定义输出： 
    out=np.zeros((1,2+10)) # 
    out[:,n+2]=1. # debug
    if z is None:
        return (out)
    else:
        for i in range(len(z)):
            out[:,i]=z[i]
        return (out)

def gen_num(num_vct):
    """
    根据指定的向量构建数字
    """
    cvae=torch.load(r"自己训练的模型\cvae模型\cvae.pt")
    decoder=cvae.decoder
    with torch.no_grad():
        num_vct=torch.FloatTensor(num_vct).to(device) # 此处tensor的类型需要是float
        de_out=decoder(num_vct)
        digit=de_out[0].reshape(28,28).cpu().numpy()
        plt.imshow(digit,cmap=plt.cm.gray)
        plt.show()

def gen_diff(num):
    """
    检验cvae输出一个数字可以有多种形式
    z1和z2的作用：
        当更改z1（在y轴上）时，数字样式变得更窄。
        改变z2的值（在x轴上）似乎会略微旋转数字，并拉长相对于上部的下部。
        这两个值之间似乎存在一些相互作用。
    """
    cvae=torch.load(r"自己训练的模型\cvae模型\cvae.pt")
    decoder=cvae.decoder
    max_z=1.5 # 这个是用来干嘛的？
    sides=8
    img_it = 0
    with torch.no_grad():
        for i in range(8):
            z1 = (((i / (sides-1)) * max_z)*2) - max_z
            for j in range(0, sides):
                z2 = (((j / (sides-1)) * max_z)*2) - max_z
                z_ = [z1, z2]
                num_vct= construct_numvec(num, z_)
                num_vct=torch.FloatTensor(num_vct).to(device) 
                de_out=decoder(num_vct)
                plt.subplot(sides, sides, 1 + img_it)
                img_it +=1
                digit=de_out[0].reshape(28,28).cpu().numpy()
                plt.imshow(digit, cmap = plt.cm.gray)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
        plt.show()

if __name__=="__main__":
    cvae=torch.load(r"自己训练的模型\cvae模型\cvae.pt")
    print(cvae)
    # 加载数据集：
    x_train,x_test,y_train,y_test=utils_wsy.load_data()
    # 将标签转为one_hot编码：
    y_train_one_hot=F.one_hot(torch.LongTensor(y_train),num_classes=10) # 手写数字识别一共有10类
    # print(y_train)
    y_test_one_hot=F.one_hot(torch.LongTensor(y_test),num_classes=10)
    # 训练(因为自己实现的是手动分批，所以此处先变回array) 
    train(x_train,y_train_one_hot.numpy()) #由于收敛比较缓慢，修改batch_size(256->64)，修改之后收敛速度差不多，所又修改回来了
    # 10(loss:168.76284790039062)
    # 20(loss:168.33079528808594)
    # 30(loss:168.40895080566406)
    # 50(loss:168.44012451171875)
    # 100(loss:168.35604858398438)
    # 测试
    test(x_test,y_test_one_hot.numpy())    
    # 10(loss: 6714.34912109375)
    # 20(6705.60400390625)
    # 30(loss: 6702.63427734375)
    # 50(loss: 6698.4833984375)
    # 100(loss: 6696.2607421875)
    # 展示编码器得到的隐变量分布
    show(x_train,y_train,y_train_one_hot)
    show(x_test,y_test,y_test_one_hot) # 此处使用的需要是标量
    # 形成要生成的数字的one_hot
    sample=construct_numvec(7)
    # 生成数字
    gen_num(sample)
    # 一个数字可能生成多种形式 
    gen_diff(2) 