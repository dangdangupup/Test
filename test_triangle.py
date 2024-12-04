import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tris

import triangle

# ------ data -------
# d = np.random.randn(100, 2)
# t = np.linspace(0, 10*np.pi)
# x = t*np.sin(t)
# y = t*np.cos(t)
# d = np.array([x, y]).T

# ------ function -------
data = triangle.get_data('A')


def plot_origin_data(data):
    vt = data['vertices']
    seg = data['segments']
    for ii in seg:
        line = vt[ii]
        plt.plot(line[:, 0], line[:, 1], c='y')
    plt.scatter(vt[:, 0], vt[:, 1])
    # plt.show()

def plot_tri(d, f, c='b'):
    for i in range(3):
        line = d[[f[i], f[(i+1)%3]]]
        print(line)
        plt.plot(line[:, 0], line[:, 1],c=c)

def plot_mesh(d, fs):
    for f in fs:
        plot_tri(d, f)

vt = data['vertices']
seg = data['segments']
plot_origin_data(data)



# hull = triangle.convex_hull(d)
# res = triangle.delaunay(vt)

res = triangle.triangulate({'vertices':data['vertices']},
                            # 'segments':data['segments']}, 
                            )
print(res)
vt = res['vertices']
res = res['triangles']
# delau = triangle.voronoi(d)




plot_mesh(vt, res)
plt.show()

# print(hull)
# for ii in hull:
#     line = np.array(d[ii])
#     plt.plot(line[:, 0], line[:, 1])

# plt.scatter(d[:, 0], d[:, 1])
# plt.show()
# plt.triplot(d, hull.tolist())