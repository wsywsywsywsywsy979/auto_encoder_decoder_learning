"""
    author:wsy
    date:2022-6-14
"""
from form_fbank_exersise import main,plot_spectrogram
from scipy.fftpack import dct
import numpy as np
if __name__=="__main__":
    while True:
        c=input( "停止程序输入-1,"+
        "计算得到mfcc光谱图输入0:")
        if c=="-1":
            break
        elif c=="0":  # 离散余弦变换，应用离散余弦变换（DCT）对滤波器组系数去相关处理(所以,需要先算fbank)
            filter_banks=main()
            plot_spectrogram(filter_banks.T, ylabel='Filter Banks')
            num_ceps = 12
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)] # 保持在2-13
            """
            scipy.fftpack.dct:返回任意类型序列x的离散余弦变换。
            """
            plot_spectrogram(mfcc.T, 'MFCC Coefficients') # 注意此处要转置，得到了光谱图
            #-------------------对fbank和mfcc进行均值化处理-------------------------------------------
            # filter_banks去均值    
            filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
            plot_spectrogram(filter_banks.T, ylabel='Filter Banks')
            #同样可以对mfcc进行去均值操作。
            mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
            plot_spectrogram(mfcc.T, 'MFCC Coefficients')


