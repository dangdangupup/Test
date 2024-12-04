import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.fft import fft, ifft
import pywt

import scipy.signal


def show_img_data():
    # plt.figure(1)
    # plt.imshow(pywt.data.camera())
    # plt.figure('ascent')
    # plt.imshow(pywt.data.ascent())
    # plt.figure('aero')
    # plt.imshow(pywt.data.aero())

    cv2.imshow("1", pywt.data.camera())
    cv2.imshow("ascent", pywt.data.ascent())
    cv2.imshow("aero", pywt.data.aero())
    cv2.waitKey(-1)

def show_singal_data():
    num = 1024
    names = pywt.data.demo_signal('list')
    for i, k in enumerate(names):
        plt.subplot(len(names), 1, i+1)
        try:
            plt.plot(pywt.data.demo_signal(k, num))
        except Exception as e:
            print(f" === {k} {e}")
            plt.plot(pywt.data.demo_signal(k))
        
        plt.title(k)

    # plt.figure(2)
    # plt.plot(pywt.data.ecg())
    # plt.title('ecg')

    
    # plt.figure(3)
    # t, s = pywt.data.nino()
    # plt.plot(t, s)
    # plt.title('nino')

    plt.show()


def generate_data():
    T = 2
    Fs = 2000
    t = np.linspace(0, T, T*Fs+1)
    data_t = np.sin(10*t)+ np.sin(100*t) + np.sin(400*t)
    return Fs, data_t

def show_fft(Fs, data_t):
    
    # Fs = 2000
    # data_t = np.sin(10*t)+ np.sin(100*t) + np.sin(400*t)
    
    data_f = np.abs(np.fft.fft(data_t))

    N = 3

    plt.subplot(N, 1, 1)
    plt.plot(data_t)


    plt.subplot(N, 1, 2)
    angle = np.linspace(0, 2*np.pi, len(data_f))
    plt.plot(angle, data_f)

    plt.subplot(N, 1, 3)
    angle = np.linspace(0, 2*np.pi*Fs, len(data_f))
    plt.plot(angle, data_f)
    plt.show()



# show_fft(*(generate_data()))

# show_fft(500, pywt.data.ecg())
# show_fft(20000, pywt.data.demo_signal('Doppler', 1000))
# show_singal_data()
names = ['Blocks', 'Bumps', 'HeaviSine', 'Doppler', 'Ramp', 'HiSine', 'LoSine', 'LinChirp', 'TwoChirp', 'QuadChirp', 'MishMash', 'WernerSorrows', 'HypChirps', 'LinChirps', 'Chirps', 'Gabor', 'sineoneoverx', 'Piece-Regular', 'Piece-Polynomial', 'Riemann']
# show_fft(20000, pywt.data.demo_signal('LinChirp', 1024))
T = 1
fs = int(10e3)
t = np.linspace(-T, T, T*fs +1)
f0 = 100.0
f1 = 1000.0
# data = np.sin(((f1-f0)*np.power(t, 0.5) + f0)*t)
# data = np.sin(((f1-f0)*t/T + f0) * t)
# data = scipy.signal.chirp(t, f0, T, f1, method='logarithmic')

# data = pywt.data.demo_signal('Gabor')
data = pywt.data.ecg()
# show_fft(10e3, data)


# plt.plot(t, np.cos(2*np.pi*3*t + np.pi/2) * np.exp(-t**2/0.1))

w = pywt.Wavelet('coif4')
d = w.wavefun()
plt.plot(d[2], d[0])
plt.plot(d[2], d[1])

plt.show()