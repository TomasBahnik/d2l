from mxnet import nd


def main():
    # data_manipulation()
    operations()


def data_manipulation():
    x = nd.arange(12)
    print('{}'.format(x))
    print('{}'.format(x.shape))
    # -1 => computes remaining dimensions
    print('{}'.format(x.reshape(-1, 4)))
    y = nd.zeros((2, 3, 4))
    print('{}'.format(y))
    z = nd.random.normal(0, 1, shape=(3, 4))
    print('{}'.format(z))


def operations():
    x = nd.array([1, 2, 4, 8])
    y = nd.ones_like(x) * 2
    print('x =', x)
    print('y =', y)
    print('x + y', x + y)
    print('x - y', x - y)
    print('x * y', x * y)
    print('x / y', x / y)


if __name__ == '__main__':
    main()
