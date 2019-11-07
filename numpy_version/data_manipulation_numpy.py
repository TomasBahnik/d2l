from mxnet import np, npx
import sys


def n_dim_array():
    x = np.arange(12)
    print(x, type(x))
    x = x.reshape(-1, 3)
    print(" x {} of type '{}' with shape '{}'".format(x, type(x), x.shape))
    y = np.zeros((2, 3, 4))
    print(" y {} with shape {}".format(y, y.shape))
    z = np.random.normal(10, 1, size=(3, 4))
    print("z {} with shape {}".format(z, z.shape))
    a = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print("a {} with shape {}".format(a, a.shape))


def n_dim_array_operations():
    x = np.array([1, 2, 4, 8])
    y = np.array([2, 2, 2, 2])
    print(x + y, x - y, x * y, x / y, x ** y)  # The ** operator is exponentiation
    print("e^x of {} = {}".format(x, np.exp(x)))
    print("sin(x) of {} = {}".format(x, np.sin(x)))

    x = np.arange(12).reshape(3, 4)
    y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    axis0 = np.concatenate([x, y], axis=0)
    print("concat axis 0 : {}, shape {}".format(axis0, axis0.shape))
    axis1 = np.concatenate([x, y], axis=1)
    print("concat axis 1 : {}, shape {}".format(axis1, axis1.shape))
    equal = x == y
    print("equal x = y: {} == {} = {}".format(x, y, equal))


def broadcast():
    a = np.arange(3).reshape(3, 1)  # row vector 3
    b = np.arange(2).reshape(1, 2)  # column vector 2
    print("a + b = {} , a + b shape {}".format(a + b, (a + b).shape))


def slicing():
    x = np.arange(12).reshape(3, 4)
    print("last {}".format(x[-1]))
    print("first two {}".format(x[0:2]))
    print("last two {}".format(x[1:-1]))


def memory():
    x = np.arange(12).reshape(3, 4)
    y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    before = id(x)
    x += y
    equals = id(x) == before
    print("id(x) after {} : id(x) before {} : equals {}".format(id(x), before, equals))

    z = np.sin(y)
    print('id(z):', id(z))
    z[:] = x + y
    print('id(z):', id(z))

    a = x.asnumpy()
    b = np.array(a)
    print("type a {}, type b {}, id(a) {}, id(b) {}".format(type(a), type(b), id(a), id(b)))

    a = np.array([3.5])
    a, a.item(), float(a), int(a)


def main():
    # n_dim_array()
    # n_dim_array_operations()
    # broadcast()
    # slicing()
    memory()


if __name__ == '__main__':
    npx.set_np()
    main()
    sys.exit(0)
