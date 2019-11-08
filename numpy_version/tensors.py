import sys
from mxnet import np, npx

npx.set_np()


def vectors():
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


def matrices(A):
    # matrix
    print("shape of matrix \n{} = {}\n".format(A, A.shape))
    # print("shape of transposed matrix \n{} = {}\n".format(A.T, A.T.shape))
    # axis=0 : move in the row direction i.e. sum columns
    A_sum_axis0 = A.sum(axis=0)
    print("shape of reduced matrix (axis=0 column sum)\n{} = {}\n".format(A_sum_axis0, A_sum_axis0.shape))
    # axis=1 : move in the column direction i.e. sum rows
    A_sum_axis1 = A.sum(axis=1)
    print("shape of reduced matrix (axis=1 row sum) \n{} = {}\n".format(A_sum_axis1, A_sum_axis1.shape))
    print(A_sum_axis0, A_sum_axis0.shape)

    # Non-Reduction Sum
    non_reduction(A, 1)
    non_reduction(A, 0)


def non_reduction(A, axis):
    sum_A = A.sum(axis=axis, keepdims=True)
    print("shape of non-reduced matrix (axis={}) \n{} = {}\n".format(axis, sum_A, sum_A.shape))
    B = A / sum_A
    print("shape of A/sum_A non-reduced matrix (axis={}) \n{} = {}\n".format(axis, B, B.shape))


def main():
    A = np.arange(20).reshape(5, 4)
    matrices(A)


if __name__ == '__main__':
    main()
    sys.exit(0)
