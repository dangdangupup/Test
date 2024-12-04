
import cv2
import numpy as np
import matplotlib.pyplot as plt
import triangle
import time




def wave_1(pos_x=0, pos_y=0):
    N = 100
    f = 1
    t = np.linspace(0, 5, N) # r
    z = np.sin(2*np.pi*f*t+pos_x) * np.exp(-0.1*t*t)

    theta = (2*np.pi*f*t).reshape(-1, 1) 
    phi = np.linspace(0, 2*np.pi, 100).reshape(1, -1)
    xs = theta * np.cos(phi)
    ys = theta * np.sin(phi)
    zs = z.reshape(-1, 1).repeat(100, axis=1)

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(xs, ys, zs)
    # ax.plot_wireframe(xs, ys, zs)
    # ax.scatter(xs, ys, zs)
    # ax.plot_trisurf(xs.reshape(-1), ys.reshape(-1), zs.reshape(-1))
    # plt.show()
    return xs + pos_x, ys + pos_y, zs

def wave_3():
    N = 100
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    r = np.sqrt(x**2 + y**2)
    # z = np.sin(r)


def wave_2(pos_x=0, pos_y=0):
    N = 1000
    B = 3
    x = np.linspace(-B, B, N) + pos_x
    y = np.linspace(-B, B, N) + pos_y
    # zs = np.sin(np.exp(x.reshape(-1, 1)**2 + y.reshape(1, -1)**2))
    L = x.reshape(-1, 1)**2 + y.reshape(1, -1)**2
    R = np.sqrt(L)
    zs = np.sin(2*np.pi*R)*np.exp(-R)
    
    xs, ys = np.meshgrid(x, y)
    return xs, ys, zs


def multi_wave_2():
    xs, ys, zs = wave_2()
    N = 100
    for i in range(1, N):
        xs_d, ys_d, zs_d = wave_2(pos_x = 0.1*i)
        xs += xs_d
        ys += ys_d
        zs += zs_d

    xs /= N
    ys /= N
    zs /= N


    ax = plt.axes(projection='3d')
    ax.plot_surface(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


# multi_wave_2()

# xs, ys, zs = wave_1(0, 0)
# M = 20
# zs_1 = np.copy(zs)
# zs_1[M:, :] = zs_1[:-M, :]
# zs_1[:M, :] = 0.0
# # xs /= M
# # ys /= M
# # zs /= M
# # zs = (zs + zs_1)/2
# zs = zs_1
# ax = plt.axes(projection='3d')
# ax.plot_surface(xs, ys, zs)
# plt.show()

theta = np.linspace(-1, 2*np.pi, 100)
rs = np.linspace(0, 1, 10)

zs_list = list()
for r in rs:
    # xs = r * np.cos(theta)
    # ys = r * np.sin(theta)
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    zs = xs + ys*1j
    zs_list.append(zs)


def keep_circle(z, z0):
    # w = (z - 0.5*(1+1j)) / (z - 0.5*(1+1j))
    return (z - z0) / (1 - np.conj(z0)*z)

ws_list = list()
for zs in zs_list:
    ws_list.append(keep_circle(zs, 0.5*(1+1j)))

plt.subplot(121)
for zs in zs_list:
    plt.plot(np.real(zs), np.imag(zs))

plt.subplot(122)
for ws in ws_list:
    plt.plot(np.real(ws), np.imag(ws))

plt.show()

