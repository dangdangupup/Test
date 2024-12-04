"""
    测试了：利用数学构造几何形状
"""
# import plotly.graph_objects as go
# import numpy as np
# import matplotlib.pyplot as plt

# fig = go.Figure()

# fig.add_trace(go.Scattersmith(
#     imag=[0],
#     real=[1],
#     marker_symbol='x',
#     marker_size=30,
#     marker_color="green",
#     subplot="smith"
# ))
 

# fig.update_layout(
#     smith = dict(
#         realaxis_gridcolor='red',
#         imaginaryaxis_gridcolor='blue',
#         domain=dict(x=[0,0.45])
#     )
# )  

# fig.update_smiths(bgcolor="lightgrey")


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure("dd")
g = 1.0

def on_key_press(event):
    global g
    
    if event.key == 'up':
        g += 0.1
    elif event.key == 'down':
        g -= 0.1
    
    print(event.key, g)
    plot_ball()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key_press)


def plot():
    global g
    u = np.linspace(0, 2*np.pi, 101).reshape(-1, 1)
    v = np.linspace(-1, 1, 101).reshape(1, -1)

    xs = v**2 * np.sqrt((1-v)/g) * np.cos(u)
    ys = v**2 * np.sqrt((1-v)/g) * np.sin(u)
    zs = np.repeat(v, u.shape[0], 0)

    ax = plt.axes(projection='3d')
    ax.plot_surface(xs, ys, zs)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

# plot()
# plt.show()

def plot_s():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)*np.sin(x/g)**g
    plt.clf()
    plt.plot(np.sin(x), label='1')
    plt.plot(np.sin(x/2)**g, label='2')
    plt.plot(y, label='3')
    plt.legend()

def plot_drop():
    global g
    u = np.linspace(0, 2*np.pi, 31).reshape(-1, 1)
    v = np.linspace(0, 2*np.pi, 31).reshape(1, -1)
    
    x = np.cos(v)
    y = np.sin(v) * np.sin(v/2)**g

    xs = x
    ys = y*np.cos(u)
    zs = y*np.sin(u)
    # z = np.repeat(z, u.shape[0], 0)
    plt.clf()
    # plt.plot(x, y)
    # plt.plot_surface(x, y, z)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xs, ys, zs)

def plot_sin():
    global g
    u = np.linspace(0, g*np.pi, 31).reshape(-1, 1)
    v = np.linspace(0, g*np.pi, 31).reshape(1, -1)
    
    x = np.sin(u)
    y = np.sin(v)
    z = np.sin(u+v)
    # z = u # np.ones((x.shape[0], x.shape[0]))
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)


def plot_mobiwu():
    v = np.linspace(0, 2*np.pi, 31).reshape(1, -1)
    
    x = [0.5 + np.cos(v/2)]*np.cos(v)
    y = [0.5 + np.cos(v/2)]*np.sin(v).reshape(-1, 1)
    z = np.sin(v/2).repeat(31, 0)
    
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)
    # ax.plot(x,y,z)


def plot_dini_fake():
    u = np.linspace(0, g*np.pi, 31).reshape(-1, 1)
    v = np.linspace(0, 0.6*np.pi, 31).reshape(1, -1)

    x = v * np.cos(u)
    y = v * np.sin(u)
    z = np.sin(v) + 0.1*u


    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)

def plot_woniu():
    u = np.linspace(0, 4*np.pi, 101)#.reshape(1, -1)
    v = np.linspace(0, 2*np.pi, 101)#.reshape(-1, 1)
    l = np.linspace(0, 2*np.pi, 101)

    cx = v*np.cos(v)
    cy = v*np.sin(v)

    # x = v*np.sin(v)*np.cos(u) 
    # y = v*np.sin(v)*np.sin(u)
    # z = np.cos(v) + g*u

    xs = v*np.sin(u)
    ys = v*np.cos(u)
    zs = v

    xyzs = np.stack([xs, ys, zs]).T

    Z = np.array([0, 0, 1])
    lx, ly = np.cos(l), np.sin(l)
    lxyzo = np.stack([lx, ly, np.zeros(len(lx)), np.ones(len(lx))])

    res_list = list()
    for i, xyz in enumerate(xyzs):
        if i == 0:
            continue

        zvt = xyz - xyzs[i-1]
        zvt = zvt / np.linalg.norm(zvt)
        # Method 1: 
        xvt = np.cross(zvt, Z)
        xvt = xvt /np.linalg.norm(xvt)
        yvt = np.cross(zvt, xvt)
        M = np.array([xvt, yvt, zvt, xyz]).T
        
        s = np.sin(xyz[-1])
        S = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],])
        res_xyz = M @  S @ lxyzo
        res_list.append(res_xyz.T)
        
    res_xyzs = np.stack(res_list)
    print(res_xyzs.shape)
        


    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(res_xyzs[:,:, 0], res_xyzs[:, :, 1], res_xyzs[:, :, 2])
    # ax.plot(x.reshape(-1), y.reshape(-1), z.reshape(-1))


def plot_egg():
    u = np.linspace(0, 5*np.pi, 101).reshape(-1, 1)
    v = np.linspace(0, 5*np.pi, 101).reshape(1, -1)

    x = u
    y = v
    z = np.sin(x) * np.sin(y + g - 1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)

def plot_langhua():
    # 浪花
    u = np.linspace(0, 4*np.pi, 101).reshape(-1, 1)
    v = np.linspace(0, 4*np.pi, 101).reshape(1, -1)
    r = 0.8
    theta = np.pi/4

 
    x = u + r*np.sin(u + g)
    y = v

    rx = x*np.cos(theta) - y*np.sin(theta)
    ry = x*np.sin(theta) + y*np.cos(theta)

    x = rx
    y = ry
    
    z = 0.1 * r*np.cos(u)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_zlim(0, 1)

def plot_zongzi():
    t = np.linspace(0, 2*np.pi, 100)
    u = np.linspace(0, np.pi, 100).reshape(1, -1)
    v = np.linspace(0, np.pi, 100).reshape(-1, 1)
    
    # x = np.sin(t)
    # y = np.sin(11*t + np.pi/2)
    # z = np.sin(10*t)

    x = np.cos(u)
    y = np.cos(v)
    z = np.cos(g*u + g*v)

    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)    

def plot_woniu_2():
    u = np.linspace(0, g*np.pi, 100).reshape(1, -1)
    v = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)

    x = u
    y = np.cos(v) + 1
    z = np.sin(v)

    xx =np.sqrt(x) * y*np.sin(x)
    yy =np.sqrt(x) * y*np.cos(x)
    zz =np.sqrt(x) * z + 0.5*x


    plt.clf()
    ax = plt.axes(projection='3d')
    # ax.plot_surface(x, y, z)    
    ax.plot_surface(xx, yy, zz)    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    

def plot_heart():
    t = np.linspace(0, 2*np.pi, 101).reshape(1, -1)
    v = np.linspace(0, g*np.pi, 101).reshape(-1, 1)

    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

    x = np.sin(v) * x
    y = np.sin(v) * y
    z = np.cos(v) * 3


    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)    
    ax.set_zlim(-15, 15)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

def plot_ball():
    u = np.linspace(0, 2*np.pi, 101).reshape(1, -1)
    v = np.linspace(0, g*np.pi, 101).reshape(-1, 1)

    x = np.cos(u)
    y = v #np.sin(v)
    z = np.sin(v) * np.sin(u)

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z)    
    ax.set_zlim(-3, 3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)


plot_ball()
plt.show()

# plt.plot_surface(x, y, z)
# print(x.shape, y.shape, z.shape)
