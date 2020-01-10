import mxnet as mx
import numpy as np
import sys
from matplotlib import pyplot as plt
from mxnet import npx, nd


def gpu_exists():
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(0))
    except:
        return False
    return True


data_ctx = mx.cpu()


def transform(data, label):
    return nd.floor(data / 128).astype('float32').squeeze(axis=-1), label


mnist = mx.test_utils.get_mnist()
# mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
# mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
train_data = np.reshape(mnist['train_data'], (-1, 28 * 28))
test_data = np.reshape(mnist['test_data'], (-1, 28 * 28))

if gpu_exists():
    print('Using GPU for model_ctx')
    model_ctx = mx.gpu(0)
else:
    print('Using CPU for model_ctx')
    model_ctx = mx.cpu()


def show_mnist():
    mx.random.seed(1)
    output_fig = False
    mnist = mx.test_utils.get_mnist()
    # print(mnist['train_data'][0].shape)
    # plt.imshow(mnist['train_data'][0][0],cmap='Greys')

    n_samples = 10
    idx = np.random.choice(len(mnist['train_data']), n_samples)
    _, axarr = plt.subplots(1, n_samples, figsize=(16, 4))
    for i, j in enumerate(idx):
        axarr[i].imshow(mnist['train_data'][j][0], cmap='Greys')
        # axarr[i].axis('off')
        axarr[i].get_xaxis().set_ticks([])
        axarr[i].get_yaxis().set_ticks([])
    plt.show()


def main():
    show_mnist()


if __name__ == '__main__':
    npx.set_np()
    main()
    sys.exit(0)
