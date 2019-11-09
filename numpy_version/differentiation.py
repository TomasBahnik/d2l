import sys
from mxnet import autograd, np, npx

npx.set_np()


def f(x):
    # return np.dot(x, x)
    # return x.sum()
    return x * x


# manually calculated grad of to f(x)
def grad_f(x):
    # return 2 * x # for dot(x,x)
    # return np.ones_like(x) # x.sum()
    # for vector x * x
    x.attach_grad()
    with autograd.record():
        y = (x * x).sum()  # v is a scalar and grad is the same as for vector x * x
    y.backward()
    return x.grad


# x : the point where the gradient
# for the scalar function f(x) is calculated
def grad_f_at(x):
    x.attach_grad()

    print("l2(x) = {}, l2^2(x) = {}".format(np.linalg.norm(x), np.linalg.norm(x) ** 2))
    with autograd.record():
        y = f(x)
    y.backward()
    print("a={}, y(x)|_a = {}".format(x, y))
    print("gradient of f in x={} : {}".format(x, x.grad))
    print("test gradient : {}".format(x.grad == grad_f(x)))


def detached(x):
    x.attach_grad()
    with autograd.record():
        y = x * x
        u = y.detach()
        z = u * x
    z.backward()
    print(x.grad == u)
    y.backward()
    print(x.grad == 2 * x)


# not analytically defined function
# where gradient can be numerically computed
def g(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


def grad_ctrl_flow():
    x = np.random.normal()
    x.attach_grad()
    with autograd.record():
        d = g(x)
    d.backward()
    print(x.grad == d / x)


def main():
    grad_ctrl_flow()


if __name__ == '__main__':
    main()
    sys.exit(0)
