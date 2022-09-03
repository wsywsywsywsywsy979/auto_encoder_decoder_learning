"""
    author:wsy
    date:2022-6-23
"""
import torch 
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
#---------------------------------------------------------------
from os import sys, path
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(path.join(PARENT_DIR,"mycode"))
import utils_wsy
#---------------------------------------------------------------
#---------------定义全局变量(定义超参数)-------------------------
bottleneck=16 # 仿照autovc设置隐层瓶颈
batch_size=64 # 定义批量大小 对于first最好的是128
epochs=50
inter_dim=32 # 此处设置和基本编码器设置一样
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu") # 设置GPU
#---------------------------------------------------------------
class vae_encoder(nn.Module):
    def __init__(self) -> None:
        super(vae_encoder,self).__init__()
        # 定义网络结构：将输入影射为隐分布的参数
        #----------first---------------------------------------------------------------
        # self.en_1=nn.Sequential(
        #     nn.Linear(784,inter_dim), # 因为还是使用手写数字识别来弄，所以输入通道为784
        #     nn.ReLU()
        # )
        #-----------------------------------------------------------------------------
        #--------------ref------------------------------------------------------------
        self.MLP = nn.Sequential()
        layer_sizes=[784, 256]
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #------------------------------------------------------------------------------
        #----------first---------------------------------------------------------------
        # self.en_mean=nn.Linear(inter_dim,bottleneck) # 拟合隐变量分布均值
        # self.en_sigma=nn.Linear(inter_dim,bottleneck) # 拟合隐变量分布的方差
        #-----------------------------------------------------------------------------
        #--------------ref------------------------------------------------------------
        latent_size=2 # 此处需要参考赋值
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #------------------------------------------------------------------------------

    def forward(self,x):               # x:(512, 784)
        #----------first---------------------------------------------------------------
        # x=self.en_1(x)                 # x:(512, 32)
        # z_mean=self.en_mean(x)         # z_mean:(512, 16)
        # z_log_sigma=self.en_sigma(x)   # z_log_sigma:(512, 16)
        # return z_mean,z_log_sigma
        #-----------------------------------------------------------------------------
        #--------------ref------------------------------------------------------------
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars
        #------------------------------------------------------------------------------

class vae_decoder(nn.Module):
    def __init__(self) -> None:
        super(vae_decoder,self).__init__()
        # 定义网络结构
        #----------first---------------------------------------------------------------
        # self.de_1=nn.Sequential(
        #     nn.Linear(bottleneck,inter_dim),
        #     nn.ReLU()
        #     )
        # self.de_mean=nn.Sequential(
        #     nn.Linear(inter_dim,784),
        #     nn.Sigmoid()
        # )
        #-----------------------------------------------------------------------------
        #--------------ref------------------------------------------------------------
        input_size=2
        layer_sizes=[256, 784]
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        #------------------------------------------------------------------------------

    def forward(self,x):
        #----------first---------------------------------------------------------------
        # x=self.de_1(x)
        # de_mean=self.de_mean(x)
        # return de_mean
        #-----------------------------------------------------------------------------
        #--------------ref------------------------------------------------------------
        x = self.MLP(x)
        return x
        #------------------------------------------------------------------------------

#---------------------对比之后新增----------------------------------------------
class vae(nn.Module):
    """
    encoder-decoder框架
    """
    def __init__(self) -> None:
        super(vae,self).__init__()
        # 定义整个框架结构
        self.encoder=vae_encoder()
        self.decoder=vae_decoder()
    
    def forward(self,data):
        z_mean,z_log_sigma=self.encoder(data)
        z=utils_wsy.sampling([z_mean,z_log_sigma])
        de_mean=self.decoder(z)
        return z_mean,z_log_sigma,de_mean
#-----------------------------------------------------------------------------------
def train(x_train):
    """
    function:训练
    para:x_train:训练集
    """
    # 创建模块对象：
    #-------------训练采用逐渐加码的方式：除第一次训练外，其余都加载之前保存的模型继续训练---
    v=torch.load("vae.pt")
    #---------------------------------------------------------------------------------
    # 设置优化器：
    opt=torch.optim.Adam(v.parameters(),lr=0.001)
    # 训练：
    size=x_train.shape[0]
    for epoch in range(epochs):
        index=0
        while index+batch_size < size:
            # 加载输入数据
            data=np.array(x_train)[index:index+batch_size,:]
            index+=batch_size
            data=Variable(torch.tensor(data)).to(device)
            # 梯度清零
            opt.zero_grad()
            z_mean,z_log_sigma,de_mean=v(data)      
            # 计算损失并回传
            loss=utils_wsy.vae_loss(data,de_mean,z_log_sigma,z_mean)
            loss.backward() 
            opt.step()    
        a=loss.item()         
        print("epoch:"+str(epoch)+":loss----"+str(a)+",index----"+str(index)) 
        print("##########################################") 
    torch.save(v,"vae.pt") 
    # # 保存训练得到的隐变量的均值和方差
    state={"z_mean":z_mean,"z_log_sigma":z_log_sigma}
    torch.save(state,"mean_sigma.pt") # 保存均值和方差
    # 开始想的是生成时需要从训练拟合的隐变量的分布中进行采样
    # 但因为隐变量所拟合的就是标准正态分布，生成时只需要从标准标准正态分布中采样作为隐变量即可

def test(x_test):
    """
    测试
    """
    v=torch.load("vae.pt")
    print(v)       
    index=0
    size=x_test.shape[0]
    loss=0
    with torch.no_grad():
        while index+batch_size < size:
            # 加载输入数据
            data=np.array(x_test)[index:index+batch_size,:]
            index+=batch_size
            data=Variable(torch.tensor(data)).to(device)
            # 前向传播：
            z_mean,z_log_sigma,de_mean=v(data)           
            # 计算损失并回传
            loss+=utils_wsy.vae_loss(data,de_mean,z_log_sigma,z_mean)
        print("loss:",loss.item())
        
def predic(x,y):
    """
    用于可视化查看编码器深层的隐式分布：根据目前的效果来看，拟合还是比较混乱（交叠较多）
    """
    v=torch.load("vae.pt")
    encoder=v.encoder
    with torch.no_grad():
        x=torch.FloatTensor(x).to(device)
        z_mean,z_log_sigma=encoder(x)
        plt.figure(figsize=(6, 6))
        z_mean=z_mean.detach().cpu().numpy()
        z_log_sigma=z_log_sigma.detach().cpu().numpy()
        # 下方参数需要是array类型，所以需要先转回来，注意三个参数的数量需要是一致的！
        plt.scatter(z_mean[:,0], z_log_sigma[:,0], c=y) 
        plt.colorbar()
        plt.show()

def gen():
    """
    使用vae生成新数字
    """
    v=torch.load("vae.pt")
    mydecoder=v.decoder 
    n=10
    state=torch.load("mean_sigma.pt")
    z_mean=state["z_mean"]
    z_log_sigma=state["z_log_sigma"]
    print(z_mean,z_log_sigma)
    figure=np.zeros((28*n,28*n)) # 因为一个数字的长宽是28
    # 设置采样点坐标
    grid_x=np.linspace(-15,15,n)
    grid_y=np.linspace(-15,15,n)
    with torch.no_grad():
        for i,yi in enumerate(grid_x):
            for j ,xi in enumerate(grid_y):
                z_s=np.array([[xi,yi]]) # 因为训练解码器是在标准正态分布上进行采样的，所以此处方差为1
                # z_s得到的只是想要采样的点的位置，感觉此处应该的准备的参数还有拟合出来的隐变量的均值和方差
                z_s=torch.FloatTensor(z_s).to(device)
                de_input=torch.randn([10, 2]).to(device)
                de_out=mydecoder(de_input)
                digit=de_out[0].reshape(28,28)
                figure[i*28:(i+1)*28,j*28:(j+1)*28]=digit.detach().cpu().numpy() # 图像填充
    plt.figure(figsize=(10,10))
    plt.imshow(figure)
    plt.show() 

if __name__=="__main__":
    x_train,x_test,y_train,y_test=utils_wsy.load_data() 
    train(x_train) # 10(loss:163.46609497070312)+10(150.82659912109375)   +10(147.3865509033203)+20(142.13101196289062)
    test(x_test)   # 10(loss: 24596.4375)       +10(loss：24063.314453125)+10(23874.68359375)   +20(23704.0703125)
    predic(x_test,y_test)
    gen()