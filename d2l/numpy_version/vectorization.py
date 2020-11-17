import sys
import d2l
import math
from mxnet import np
from matplotlib import pyplot as plt

n = 10000
a = np.ones(n)
b = np.ones(n)
x = np.arange(-7, 7, 0.01)


def speed():
    timer = d2l.Timer()
    c = np.zeros(n)
    for i in range(n):
        c[i] = a[i] + b[i]
    t1 = float(timer.stop())

    timer.start()
    d = a + b
    t2 = float(timer.stop())
    print("d size {}".format(d.size))
    print('for loop : %.5f sec' % t1)
    print('vectors  : %.10f sec' % t2)
    if t2 > 0:
        print('ratio : %.1f' % float(t1 / t2))


def normal(z, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(- 0.5 / sigma ** 2 * (z - mu) ** 2)


def plot_normal():
    # Mean and variance pairs
    parameters = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in parameters], xlabel='z',
             ylabel='p(z)', figsize=(4.5, 2.5),
             legend=['mean %d, var %d' % (mu, sigma) for mu, sigma in parameters])
    plt.show()


def main():
    speed()


if __name__ == '__main__':
    main()
    sys.exit(0)
