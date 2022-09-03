"""
    sequence to sequence encoder
    author:wsy
    date:2022-6-22
    参考：https://blog.csdn.net/zhaojc1995/article/details/105596458
    效果不太好,保存了模型
"""
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import jieba
import numpy as np
import random
import itertools
# 定义词库：词库中包含单词和索引之间的映射
class voc:
    def __init__(self,name):
        """
        para:name:词库名称
        """
        self.name=name
        self.word2index={}
        self.index2word=["PAD","/t","/n"]  # "PAD" : ,"/t": 开头,"/n" ：结尾
        self.n_words = 3 # 计数词库中已有单词数
    def addword(self,word): 
        """
        function:将建立单词和id之间的映射,并添加到词库中
        para:word：要加入的单词
        """
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.index2word.append(word) # list 添加
            self.n_words+=1
#-------设置全局变量---------------------------------------------
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu") # 设置GPU
batch_size = 64 # 设置批量大小，当数据量较少时，需要设置小一些
hid_size=256
max_token_len=0 # 统计最长的长度
PAD=0
SOS_token=1
EOS_token=2
english=voc('英语') # 英语词库
chinese=voc('普通话') # 中文词库
#---增加下面这个后，在变量放置在cuda上执行时，还可定位代码出错行
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#----------------------------------------------------------------
class myencoder(nn.Module):
    """
    编码器
    """
    def __init__(self,embedding):
        """
        para:embedding: 词向量
        """
        super(myencoder,self).__init__() # 如果此处不加self,会报错：cannot assign module before Module.__init__() call
        # 定义超参数：
        self.hidden_nums = hid_size # 定义隐藏层的维度
        # 定义网络结构：
        self.embedding = embedding # 需要根据词库中的单词数量进行设置
        self.gru = nn.GRU(self.hidden_nums,self.hidden_nums) # 定义了一个循环神经网络  参数：词向量维度，隐藏层维度
    def forward(self,x,h,input_lens):
        """
        para:
            x:样本
            h:初始的隐藏层
            input_lens:batch中每个样本的原始长度组成的list
        """
        x = self.embedding(x) # (15, 64, 256)
        """
            下方语句：调整输入序列的长度  函数功能：填充一批可变长度序列。
            运行后x.shape: (483, 256)
        """
        x = nn.utils.rnn.pack_padded_sequence(x,input_lens) 
        y,h = self.gru(x,h) # debug: y:(483, 256) h:(1, 64, 256)
        """
        out的输出维度：[seq_len,batch_size,output_dim]
        ht的维度：[num_layers * num_directions, batch_size, hidden_size],num_directions=1，单向，取值2时为双向，num_layers为层数
        out[-1]=ht[-1]
        """ 
        y,_ = nn.utils.rnn.pad_packed_sequence(y) # 还原 y:(15, 64, 256)
        return y,h 

class mydecoder(nn.Module):
    """
    解码器 
    """
    def __init__(self,embedding) :
        """
        para:embedding: 词向量
        """
        super(mydecoder,self).__init__()
        # 定义超参数：
        self.hidden_nums = hid_size # 定义解码器的隐藏层维度
        self.out_size=len(chinese.index2word) # 定义输出向量的大小，因为输出是翻译为中文
        # 定义网络结构：
        self.embedding=embedding
        self.gru=nn.GRU(self.hidden_nums,self.hidden_nums)
        self.out=nn.Sequential(
            nn.Linear(self.hidden_nums,self.out_size),
            nn.Softmax(dim=1) # dim=1 让每一行和为1,因为输出的是一个序列
            )
        # 此处网络结构可优化 
    def forward(self,x,h): # x: (1, 64)
        x=self.embedding(x) # x:(1, 64, 256)
        x=F.relu(x) # x:(1, 64, 256)
        y,h=self.gru(x,h) # # y:(1, 64, 256) h:(1, 64, 256)
        y=self.out(y[0]) # 为啥这里是取y[0],因为y只有一个值，但是list元素存放
        return y, h # 返回解码器的两个输出

#将sentence转换成index
def sentence2index_eng(sentence):
    return [english.word2index[word] for word in sentence.split()] + [EOS_token]
def sentence2index_chi(sentence):
    return [chinese.word2index[word] for word in jieba.lcut(sentence)] + [EOS_token]
def binaryMatrix(index_batch):
    """
    function:构建mask矩阵
    para:index_batch：将一个batch中的输入和输出拆分后分别根据词库将每个样本中的每个词映射为id
    """
    m=[]
    for i ,seq in enumerate(index_batch):
        m.append([])
        for token in seq:
            if token==PAD:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def zeroPadding(l):
    """
    function:padding函数，用于对不等长的句子进行填充，返回填充后的转置
    """
    return list(itertools.zip_longest(*l, fillvalue=0))
    """
    itertools.zip_longest:当有可迭代对象遍历完，但其他对象还没有的时候，缺少的相应元素就会使用填充值进行填充。
    """

def data_preproceing(pair_batch):
    """
    function:数据预处理
    para:pair_batch:[["inputs","target"]]
    """
    pair_batch.sort(key=lambda x:len(x[0].split())) # 按照英文句子的长度进行升序排序
    pair_batch.reverse()
    # 将inputs 和 target 进行拆分
    input_batch,output_batch=[],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # 处理input_batch
    ## 将输入中的每个句子通过对应词库变为一个向量
    index_batch=[sentence2index_eng(s) for s in input_batch ]
    ## 记录下原始每个样本的长度!!!
    lengths=torch.tensor([len(index) for index in index_batch])
    ## 对齐，并转为tensor
    # padvar=torch.LongTensor(zeropadding) 但encoder中已经进行填充了，此处还需要填充吗？ 需要，因为变为tensor需要对齐
    padvar_input=torch.LongTensor(zeroPadding(index_batch)) # 需要填充
    # 处理out_batch:
    ## 将每一个label根据词库变为向量：
    index_batch=[sentence2index_chi(s) for s in output_batch]
    max_label_len=max([len(index) for index in index_batch]) ## 统计label的最长长度，应该是用于解码器的输出
    tmp=zeroPadding(index_batch)
    mask=binaryMatrix(tmp)  # 形成mask前就需要进行对齐
    mask=torch.BoolTensor(mask) # 只有0,1
    mask=mask.to(device) # add 
    padvar_out=torch.LongTensor(tmp)
    return padvar_input,lengths,padvar_out,mask,max_label_len 
    """
    padvar_input:shape(15, 64),max=11976,min=0
    padvar_out:shape(15, 64),max=15503,min=0
    """
     
def load_data():
    """
    function:加载数据集，并转化为['inputs','targets']格式
    """
    # 读入数据建立词库：
    df=pd.read_table(filepath_or_buffer=r'已看论文\mycode\cmn-eng\cmn.txt',header=None).iloc[:,:] # 读取制表格式的文件，并设置无head 
    df.reset_index(drop=True,inplace=True) #?
    df.columns=['inputs','targets','useless'] # 由于种种报错,此处给不使用的第三列也加上title
    df_pair=[]
    global max_token_len
    for i in range(len(df['inputs'])):
        eng=df['inputs'][i].split() # 英文直接通过空格切分即可
        chi=jieba.lcut(df['targets'][i]) # 注意细节：加s
        chi_len=len(chi)
        max_token_len=max(max_token_len,chi_len)
        for tmp in eng:
            english.addword(tmp)
        for tmp in chi:
            chinese.addword(tmp)
        df_pair.append([df['inputs'][i],df['targets'][i]]) # 保存对应翻译 
    return df_pair # 最终词库中数量：en:12502  ch:16089

def maskNLLLoss(de_out,index,mask): # de_out:(64, 25),index:(64,),mask:(64,)
    """
    对loss 进行mask操作
    para:
        de_out : 解码器的输出
        index : 要抽取的数据的索引
        mask : 掩码矩阵
    """
    # 求交叉熵
    cross_entropy=-torch.log(torch.gather(de_out,1,index.view(-1,1)).squeeze(1))  #  index.view(-1,1):(64,1) 
    # 保留 mask 中值为1的部分，并且求均值
    loss=cross_entropy.masked_select(mask).mean()
    loss=loss.to(device) # 将设备放到cuda上
    return loss 

def split_train_and_test(df_pair):
    """
    划分训练集和测试集（10折交叉运算）
    """
    kfold=KFold(n_splits=10,shuffle=False,random_state=random.seed(2022))
    for train_index,test_index in kfold.split([ i for i in range(len(df_pair))]):
        train_pair=np.array(df_pair)[train_index].tolist() # only integer scalar arrays can be converted to a scalar index 解决：将list索引前变为array
        test_pair=np.array(df_pair)[test_index].tolist()
        return train_pair, test_pair

def train(train_pair):
    """
    function:训练
    para:train_pair:训练集对
    """
    # 设置训练的超参数：
    lr = 0.001
    iters = 2000
    embedding_eng=torch.nn.Embedding(len(english.index2word),hid_size)
    embedding_ch=torch.nn.Embedding(len(chinese.index2word),hid_size) # 其实感觉embedding也是一种全连接层，para:输入通道，输出通道

    # 设置网络模块：
    encoder=myencoder(embedding_eng).to(device)
    decoder=mydecoder(embedding_ch).to(device)
    encoder_opt=torch.optim.Adam(encoder.parameters(),lr=lr)
    decoder_opt=torch.optim.Adam(decoder.parameters(),lr=lr)

    for it in range(iters):
        train_pair,test_pair=split_train_and_test(df_pair) # 划分训练集和测试集
        global max_label_len
        x,lengths,label,mask,max_label_len=data_preproceing([random.choice(train_pair) for _ in range(batch_size)])
        # 梯度归零：（优化器归零）
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        # 将数据放置到cuda上
        x=x.to(device)
        label=label.to(device)

        # 构建编码器的隐藏层：# 是把参数权重全部初始化为0
        en_hid=torch.zeros(1,batch_size,encoder.hidden_nums,device=device)  # debug: shape:(1, 64, 256)

        # 编码器前向传播：
        _,en_hid=encoder(x,en_hid,lengths) # 

        # 解码器前向传播：
        ## 构建解码器的输入？？
        de_input=torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device) # 给每个样本的都构建一个起始符
        de_hid=en_hid # 将编码器的输出作为解码器的输入

        loss=0
        for i in range(max_label_len):
            de_out,de_hid=decoder(de_input,de_hid) # de_out:(64, 25), 
            topv,topi=de_out.topk(1) # 获取sigmoid中最大的, topv:(64, 1) , topi:(64, 1)
            de_input=torch.LongTensor([[topi[j][0] for j in range(batch_size)]]).to(device) # 获取每个样本中的最大概率
            # 将解码器的输出中不重要的进行mask
            # mask[i].to(device) 注意这是一个不是在源对象上修改的函数,需要赋值为新对象
            # mask_loss=maskNLLLoss(de_out.detach().cpu(),label[i].detach().cpu(),mask[i])
            mask_loss=maskNLLLoss(de_out,label[i],mask[i])
            loss+=mask_loss

        loss.backward() # 损失回传
        # 更新参数
        encoder_opt.step() 
        decoder_opt.step() 
        if it% 100==0:
            print("iter:",it,"loss:",loss.item())
    torch.save(encoder,"se2se_encoder.pt")
    torch.save(decoder,"se2se_decoder.pt")

def predict(test_pair):
    """
    function:测试集测试
    para:test_pair:测试集对
    """
    encoder=torch.load("se2se_encoder.pt")
    decoder=torch.load("se2se_decoder.pt")
    for s in test_pair: # 此处的s是一个['英文'，'中文']
        # 将测试句子变为tensor:
        x=torch.LongTensor([sentence2index_eng(s[0])]).view(-1,1).to(device)
        res=''
        # 和训练一样，构建编码器的隐藏层
        en_hid=torch.zeros(1,1,encoder.hidden_nums,device=device).to(device) # 但一次只翻译一句
        length=torch.LongTensor([len(x)]) # 
        _,en_hid=encoder(x,en_hid,length) # 参数:x ,初始隐藏层，输入长度(注意都需要是tensor)
        de_x=torch.LongTensor([[SOS_token]]).to(device) # 解码器的输入，初始化开始符号
        de_hid=en_hid 
        for i in range(max_token_len):
            de_out,de_hid=decoder(de_x,de_hid)
            topv,topi=de_out.topk(1)
            # 因为只翻译一句，所以此处直接转为tensor
            de_x=torch.LongTensor([[topi.item()]]).to(device) # 嵌套两层是为了适配网络结构
            word=chinese.index2word[topi.item()] # 翻译
            if word=="/n": # 结束符
                 break 
            res+=word
        print("原句：",s,"翻译结果：",res)


if __name__=="__main__":
    df_pair=load_data() # 构建词库，并获取翻译对
    train_pair,test_pair=split_train_and_test(df_pair) # 划分训练集和测试集
    train(train_pair) # 训练 目前打算先训练2000次
    predict(test_pair) # 验证 