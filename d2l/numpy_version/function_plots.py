import sys
import d2l
from mxnet import np, npx
from matplotlib import pyplot as plt

npx.set_np()
d2l.use_svg_display()


def f(x):
    return 3 * x ** 2 - 4 * x


def function_plot():
    x = np.arange(0, 5, 0.1)
    d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    plt.show()


def main():
    function_plot()


if __name__ == '__main__':
    main()
    sys.exit(0)
