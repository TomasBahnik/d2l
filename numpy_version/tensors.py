import sys
from mxnet import np, npx

npx.set_np()


def main():
    # scalars are also ndarrays
    x = np.array(3.0)
    y = np.array(2.0)
    print("shape of scalar = {}".format(x.shape))
    print("size of scalar = {}".format(x.size))
    # vector
    xv = np.arange(4)
    print("shape of vector {} = {}".format(xv, xv.shape))
    print("size of vector = {}".format(xv.size))
    print(x + y, x * y, x / y, x ** y)
    # matrix
    A = np.arange(20).reshape(5, 4)
    print("shape of matrix {} = {}".format(A, A.shape))
    print("shape of transposed matrix {} = {}".format(A.T, A.T.shape))
    A_sum_axis0 = A.sum(axis=0)
    A_sum_axis1 = A.sum(axis=1)
    print(A_sum_axis0, A_sum_axis0.shape)
    # Non-Reduction Sum
    sum_A = A.sum(axis=1, keepdims=True)
    print("shape of non-reduced matrix {} = {}".format(sum_A, sum_A.shape))


if __name__ == '__main__':
    main()
    sys.exit(0)
