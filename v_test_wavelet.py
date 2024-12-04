
#使用python实现小波变换

import numpy as np
import pywt
import matplotlib.pyplot as plt

# 生成信号变量
t = np.linspace(0, 1, num=1000)
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(3 * np.pi * 30 * t)
# signal = pywt.data.ecg()
# t = np.linspace(0, 1, num=len(signal))

# 添加随机噪声
noise = np.random.normal(0, 0.05, len(signal))
signal = signal + noise

# 常见的几种小波基函数包括： 
 
# 1. Daubechies小波基（db）：Daubechies小波基是最常用的小波基函数之一。它具有紧凑支持和良好的频率局部化特性。常见的Daubechies小波基包括db2、db4、db6等。 
 
# 2. Symlets小波基（sym）：Symlets小波基是对称的Daubechies小波基。它们在频率局部化和相位对称性方面与Daubechies小波基类似。常见的Symlets小波基包括sym2、sym4、sym8等。 
 
# 3. Coiflets小波基（coif）：Coiflets小波基是具有紧凑支持和较好频率局部化特性的小波基。它们在一些应用中比Daubechies小波基具有更好的性能。常见的Coiflets小波基包括coif1、coif2、coif3等。 
 
# 4. Biorthogonal小波基（bior）：Biorthogonal小波基是一组成对的小波基函数。它们具有可变的支持长度和频率响应。常见的Biorthogonal小波基包括bior2.2、bior3.3、bior6.8等。 

wavelet_name = 'db4'  # 定义小波基名称为'db4'
# wavelet_name = 'sym4'  # 定义小波基名称为'sym4'
# wavelet_name = 'bior3.3'  # 定义小波基名称为'bior3.3'
# wavelet_name = 'haar'  # 定义小波基名称为'bior3.3'

# 小波变换
coeffs = pywt.wavedec(signal, wavelet_name, level=4)  # 使用指定小波基进行4级小波分解

# 绘制原始信号图像
plt.figure(figsize=(8, 6))
plt.subplot(5, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 绘制小波分解信号图像
for i in range(1, len(coeffs)):
    plt.subplot(5, 1, i+1)
    plt.plot(t[:len(coeffs[i])], coeffs[i])
    plt.title(f'Wavelet Coefficients - Level {i}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()