import sys
from mxnet import autograd, np, npx

npx.set_np()

# the point where the gradient is calculated
# for the scalar function y = 2 * np.dot(x, x) i.e. 2 * l_2(x)^2
# the gradient is (vector) 4x
x = np.arange(4)
x.attach_grad()


def gradient():
    print("l2(x) = {}, l2^2(x) = {}".format(np.linalg.norm(x), np.linalg.norm(x) ** 2))
    with autograd.record():
        y = 2 * np.dot(x, x)
    y.backward()
    print("a={}, y(x)|_a = {}".format(x, y))
    print("gradient of y(x) = 2 * np.dot(x, x) in x={} : {}".format(x, x.grad))


def main():
    gradient()


if __name__ == '__main__':
    main()
    sys.exit(0)
