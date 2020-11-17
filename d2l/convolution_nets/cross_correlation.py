import d2l
from mxnet import npx, np
from mxnet.gluon import nn

npx.set_np()


def convolution_test():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = np.array([[0, 1], [2, 3]])
    corr = d2l.corr2d(X, K)
    print("shape {}, value {}".format(corr.shape, corr))


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return d2l.corr2d(x, self.weight.data()) + self.bias.data()


def sample_black_white_image():
    X = np.ones((6, 8))
    X[:, 2:6] = 0
    return X


# kernel
K = np.array([[1, -1]])

Y = d2l.corr2d(sample_black_white_image(), K)
print("shape {}, value {}".format(Y.shape, Y))

# transposed
Y_t = d2l.corr2d(sample_black_white_image().T, K)
print("transposed : shape {}, value {}".format(Y_t.shape, Y_t))
