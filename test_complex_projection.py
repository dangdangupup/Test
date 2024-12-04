import matplotlib.pyplot as plt
import numpy as np

def z_circle(R=1.0):
    theta = np.linspace(0, 2*np.pi, 100)
    xs = R*np.cos(theta)
    ys = R*np.sin(theta)
    return xs + ys * 1.0j
    # return np.array([xs, ys]).T

# def show(Dxy, Txy):
#     sub = plt.subplot(1, 2, 1)
#     sub.plot(Dxy[:, 0], Dxy[:, 1])
    
#     sub = plt.subplot(1, 2, 2)
#     sub.plot(Txy[:, 0], Txy[:, 1])

def show_multi(Dzs, Tzs):
    sub = plt.subplot(1, 2, 1)
    if isinstance(Dzs, list):
        for v in Dzs:
            sub.plot(np.real(v), np.imag(v))
    else:
        sub.plot(np.real(Dzs), np.imag(Dzs))
        
    sub = plt.subplot(1, 2, 2)
    if isinstance(Tzs, list):
        for v in Tzs:
            sub.plot(np.real(v), np.imag(v))
    else:
        sub.plot(np.real(Tzs), np.imag(Tzs))

        
def show_multi_sync(Dzs, Tzs):
    sub1 = plt.subplot(1, 2, 1)
    sub2 = plt.subplot(1, 2, 2)
    if isinstance(Dzs, list):
        colors = ['b', 'r', 'g', 'm', 'y', 'k']
        for k, (d, t) in enumerate(zip(Dzs, Tzs)):

            d_x = np.real(d); d_y = np.imag(d)
            t_x = np.real(t); t_y = np.imag(t)
            for n in range(len(d_x)-1):
                plt.ion()
                sub1.plot(d_x[n:n+2], d_y[n:n+2], color=colors[k])
                sub2.plot(t_x[n:n+2], t_y[n:n+2], color=colors[k])
                plt.show()
                plt.pause(0.1)

            
    else:
        sub1.plot(np.real(Dzs), np.imag(Dzs))
        sub2.plot(np.real(Tzs), np.imag(Tzs))

    plt.ioff()

        
    
class Test:
    @classmethod
    def test_show(cls):
        c1 = z_circle(1)
        c2 = z_circle(2)
        c3 = z_circle(3)
        cs = [c1, c2]
        cs_t = [c1, c2, c3]
        show_multi(cs, cs_t)
    
    @classmethod
    def test_cos(cls):
        ds = list(); ts = list()
        for r in range(1, 6):
            d = z_circle(r)
            t = np.cos(d)
            
            ds.append(d)
            ts.append(t)
        show_multi_sync(ds, ts)
        show_multi(ds, ts)

    @classmethod
    def run(cls):
        # cls.test_show()
        cls.test_cos()
        


if __name__ == '__main__':

    Test.run()
    plt.show()






