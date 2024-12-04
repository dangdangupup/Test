import numpy as np
import matplotlib.pyplot as plt

N = 50

t = np.linspace(0, 1, N)
x = np.cos(2*np.pi*t)
y = np.sin(2*np.pi*t)
xys = np.array([x, y]).T

M = np.array([[1, -3], [1, 1]])

F_l, F_v = np.linalg.eig(M)

print('F_l', F_l)
print('F_v', F_v)

out_xys = M @ xys.T
out_xys = out_xys.T

plt.scatter(xys[:, 0], xys[:, 1], c = 'r')
plt.scatter(out_xys[:, 0], out_xys[:, 1], c = 'b')

plt.plot(xys[:, 0], xys[:, 1], c = 'r')
plt.plot(out_xys[:, 0], out_xys[:, 1], c = 'b')
plt.plot([0, F_v[0, 0]], [0, F_v[1, 0]], label='fv1', c='r', linewidth=3)
plt.plot([0, F_v[0, 1]], [0, F_v[1, 1]], label='fv2', c='r', linewidth=3)
plt.legend()


plt.figure(1)
for ii, (aa, bb) in enumerate(zip(xys, out_xys)):
    angle_aa = np.arctan2(np.abs(aa[1]), aa[0])
    angle_bb = np.arctan2(np.abs(bb[1]), bb[0])


    ab = np.array([aa, bb])
    plt.plot(ab[:, 0], ab[:, 1], c='gray')
    plt.plot([0, aa[0]], [0, aa[1]], c='black')
    # plt.text(aa[0], aa[1], f"{ii} + {angle_aa:.3f}")
    
    plt.text(bb[0], bb[1], f"{angle_bb:.3f}")

angle_aas = np.arctan2(np.abs(xys[:, 1]), xys[:, 0]) 
angle_bbs = np.arctan2(np.abs(out_xys[:, 1]), out_xys[:, 0])
plt.figure(2)
plt.plot(2*np.pi*t, angle_aas)
plt.plot(2*np.pi*t, angle_bbs)

da = angle_aas[1:] - angle_aas[:-1]
db = angle_bbs[1:] - angle_bbs[:-1]
print(f"sum a angle {sum(np.abs(da))}")
print(f"sum b angle {sum(np.abs(db))}")


# plt.figure(1)
# angle_aa = list()
# angle_bb = list()
# for ii, (aa, bb) in enumerate(zip(xys, out_xys)):
#     angle_aa.append()
    
    

plt.show()