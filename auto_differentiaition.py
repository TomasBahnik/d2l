from mxnet import autograd, nd

x = nd.arange(4)
x.attach_grad()


def main():
    # scalar()
    # vector()
    detach()


def detach():
    with autograd.record():
        y = x * x
        print('x =', x)
        print('y =x * x = ', y)
        u = y.detach()
        print('u =', u)
        z = u * x
        print('z =', z)
    z.backward()
    print('x.grad  =', x.grad)
    print('u  =', u)


def vector():
    with autograd.record():  # y is a vector
        y = x * x
    print('y =', y)
    y.backward()

    u = x.copy()
    u.attach_grad()
    with autograd.record():  # v is scalar
        v = (u * u).sum()
    v.backward()
    print('x.grad - u.grad =', x.grad - u.grad)


def scalar():
    with autograd.record():
        # y = sum 2 * x_i^2, gradient is a vector with components dy/dx_j ie. [4x_i] taken at given value
        y = 2 * nd.dot(x, x)
    print('{}'.format(y))
    y.backward()
    print('gradient of  2*x^2 at {} is  : {}'.format(x, x.grad))

    with autograd.record():
        y = x.norm()
        print('norm of {} is  : {}'.format(x, x.norm()))
    y.backward()
    print('gradient of x.norm() = x_j/x.norm(). Value at {} is {}'.format(x, x.grad))


if __name__ == '__main__':
    main()
