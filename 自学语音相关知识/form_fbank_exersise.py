"""
    author:wsy
    date:2022-6-13
"""
# fbank
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def plot_time(sig, fs,graph_name):
    # function 绘制时域图
    time = np.arange(0, len(sig)) * (1.0 / fs)
    plt.figure(figsize=(20, 5))
    plt.plot(time, sig)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude' ) # 振幅
    plt.grid()
    plt.title(graph_name)

def plot_freq(sig, sample_rate, nfft=512,graph_name=""):    
    # function:绘制频域图
    freqs = np.linspace(0, sample_rate/2, nfft//2 + 1) # np.linspace()在线性空间中以均匀步长生成数字序列。
    xf = np.fft.rfft(sig, nfft) / nfft
    """
    np.fft.rfft:
        计算实际输入的一维离散傅立叶变换。
        这个函数计算一维 n -通过一种称为快速傅立叶变换（FFT）的高效算法，对实值阵列进行点离散傅立叶变换（DFT）。
    """
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100)) # 强度
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB') # 强度
    plt.grid()
    plt.title(graph_name)

def plot_spectrogram(spec, ylabel = 'ylabel'):
    # function:绘制二维数组
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    """
    plt.pcolor专门用于画二维数组的，pcolor就是以颜色的深浅把每个位置的值表示出来，直观地展示了原数组的数据大小分布。
    """
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def framing(frame_len_s, frame_shift_s, fs, sig):
    """
    function:分帧，主要是计算对应下标
    para：
        frame_len_s: 帧长，s
        frame_shift_s: 帧移，s
        fs: 采样率，hz
        sig: 信号
    return: 二维list，一个元素为一帧信号
    """
    sig_n = len(sig)
    frame_len_n, frame_shift_n = int(round(fs * frame_len_s)), int(round(fs * frame_shift_s))
    num_frame = int(np.ceil(float(sig_n - frame_len_n) / frame_shift_n) + 1)
    pad_num = frame_shift_n * (num_frame - 1) + frame_len_n - sig_n # 待补0的个数
    pad_zero = np.zeros(int(pad_num)) # 补0
    pad_sig = np.append(sig, pad_zero)
    # 计算下标
    # 每个帧的内部下标
    frame_inner_index = np.arange(0, frame_len_n)
    # 分帧后的信号每个帧的起始下标
    frame_index = np.arange(0, num_frame) * frame_shift_n
    # 复制每个帧的内部下标，信号有多少帧，就复制多少个，在行方向上进行复制
    frame_inner_index_extend = np.tile(frame_inner_index, (num_frame, 1)) # 把数组沿各个方向复制
    # 各帧起始下标扩展维度，便于后续相加
    frame_index_extend = np.expand_dims(frame_index, 1)
    """
    查询到的解释：
        expand_dims(a, axis)中，a为numpy数组，axis为需添加维度的轴，a.shape将在该轴显示为1，通过索引调用a中元素时，
        该轴对应的索引一直为0。
    自己的理解：shape元组从0索引（和axis对应）开始，如果再axis位置增加维度，则将该维度向后推一个，然后在该维度赋值为1
    """
    # 分帧后各帧的下标，二维数组，一个元素为一帧的下标
    each_frame_index = frame_inner_index_extend + frame_index_extend
    each_frame_index = each_frame_index.astype(np.int, copy=False)
    frame_sig = pad_sig[each_frame_index]
    return frame_sig

def stft(frame_sig, nfft=512):    
    """
    para：
        frame_sig: 分帧后的信号
        nfft: fft点数
    return: 返回分帧信号的功率谱
    np.fft.fft vs np.fft.rfft
    fft 返回 nfft
    rfft 返回 nfft // 2 + 1，即rfft仅返回有效部分
    """
    frame_spec = np.fft.rfft(frame_sig, nfft)
    # 幅度谱
    frame_mag = np.abs(frame_spec) # 语音信号频谱取模
    # 功率谱
    frame_pow = (frame_mag ** 2) * 1.0 / nfft # 取平方
    return frame_pow

def mel_filter(frame_pow, fs, n_filter, nfft):    
    """
    function:mel 滤波器系数计算
    para:
        frame_pow: 分帧信号功率谱
        fs: 采样率 hz
        n_filter: 滤波器个数
        nfft: fft点数
    return: 分帧信号功率谱mel滤波后的值的对数值
    mel = 2595 * log10(1 + f/700) # 频率到mel值映射
    f = 700 * (10^(m/2595) - 1 # mel值到频率映射
    上述过程本质上是对频率f对数化
    """
    mel_min = 0 # 最低mel值
    mel_max = 2595 * np.log10(1 + fs / 2.0 / 700) # 最高mel值，最大信号频率为 fs/2
    mel_points = np.linspace(mel_min, mel_max, n_filter + 2) # n_filter个mel值均匀分布与最低与最高mel值之间
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1) # mel值对应回频率点，频率间隔指数化
    filter_edge = np.floor(hz_points * (nfft + 1) / fs) # 对应到fft的点数比例上
    # 求mel滤波器系数
    fbank = np.zeros((n_filter, int(nfft / 2 + 1)))
    for m in range(1, 1 + n_filter):
        f_left = int(filter_edge[m - 1]) # 左边界点
        f_center = int(filter_edge[m]) # 中心点
        f_right = int(filter_edge[m + 1]) # 右边界点
        for k in range(f_left, f_center):
            fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    # mel 滤波
    # [num_frame, nfft/2 + 1] * [nfft/2 + 1, n_filter] = [num_frame, n_filter]
    filter_banks = np.dot(frame_pow, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # 取对数
    filter_banks = 20 * np.log10(filter_banks) # dB
    return filter_banks

def main():
    wav_file = r'E:\repository\auto_encoder_decoder_learning\自学语音相关知识\data\p979_015_mic.wav'
    fs, sig = wavfile.read(wav_file)# fs是wav文件的采样率，signal是wav文件的内容，filename是要读取的音频文件的路径
    sig = sig[0: int(10 * fs)] # 保留前10s数据
    while True:
        # c=input( # 将长字符串放置在多行
        #     "停止程序输入-1，"+
        #     "查看原始音频的时域图和频域图输入0,"+
        #     "查看进行预加重输入1,"+
        #     "分帧处理输入2,"+
        #     "加窗输入3(需要在分帧后面执行),"+
        #     "傅里叶变换输入4(在分帧加窗后执行),"+
        #     "计算功率谱输入5(在傅里叶变换之后):")
        c=input( "停止程序输入-1,"+
        "开始处理音频输入0:")
        # "mel滤波器组输入6:")
        if c=="-1":
            break
        elif c=="0":   # 查看原始wav文件的时域图和频域图  
            plot_time(sig, fs,graph_name="original time graph")
            plot_freq(sig, fs,graph_name="original freq graph")
        # elif c=="1":  # 预加重属于1
            pre_emphasis = 0.97
            sig = np.append(sig[0], sig[1:] - pre_emphasis * sig[:-1]) # y(t)=x(t)-ax(t-1)
            """
            np.append():为原始array(第一个参数)添加一些values(第二个参数)
            """
            plot_time(sig, fs,graph_name="add weight time graph")
            plot_freq(sig, fs,graph_name="add weight freq graph")
            print(sig.shape,fs)
        # elif c=="2": # 分帧
            frame_len_s = 0.025
            frame_shift_s = 0.01
            frame_sig = framing(frame_len_s, frame_shift_s, fs, sig)
        # elif c=="3": # 加窗，需要在分帧后面，效果：加窗之后曲线变得光滑了
            window = np.hamming(int(round(frame_len_s * fs))) # 创建汉明窗口
            plt.figure(figsize=(20, 5))
            plt.plot(window)
            plt.grid()
            plt.xlim(0, 200)
            plt.ylim(0, 1)
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.title("hamming window")
            frame_sig *= window
            plot_time(frame_sig, fs,graph_name="add windows time graph")
            plot_freq(frame_sig.reshape(-1,), fs,graph_name="add windows freq graph") #需要先变成一维数据
            print(frame_sig.reshape(-1,).shape)
        # elif c=="4": # 傅里叶变换,需在加窗之后
            NFFT=512 # NFFT常为256或512
            # mag_frames = np.absolute(np.fft.rfft(fs, NFFT)) # fft的幅度(magnitude) fs=16000是帧频
            mag_frames = np.absolute(np.fft.rfft(frame_sig, NFFT)) # frame_sig 分帧之后每一帧的信号
        # elif c=="5": # 绘制功率图，需在傅里叶变换之后
            nfft = 512
            frame_pow = stft(frame_sig, nfft) # 短时傅里叶变换将帧信号变为帧功率
            plt.figure(figsize=(20, 5))
            plt.plot(frame_pow[1])
            plt.grid()  
            plt.title("power graph")
        # elif c=="6": # 使用的功率图，所以需要先得到功率图
            # mel 滤波
            n_filter = 40 # mel滤波器个数   
            filter_banks = mel_filter(frame_pow, fs, n_filter, nfft)
            plot_spectrogram(filter_banks.T, ylabel='Filter Banks') # 光谱图！！！
            return filter_banks # 因为形成mfcc需要

if __name__=="__main__":
    main()