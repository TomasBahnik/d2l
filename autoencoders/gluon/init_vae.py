import mxnet as mx
import numpy as np
import sys
from matplotlib import pyplot as plt
from mxnet import npx

batch_size = 100
mnist = mx.test_utils.get_mnist()
train_data = np.reshape(mnist['train_data'], (-1, 28 * 28))
test_data = np.reshape(mnist['test_data'], (-1, 28 * 28))
train_iter = mx.io.NDArrayIter(data={'data': train_data}, label={'label': mnist['train_label']},
                               batch_size=batch_size)
test_iter = mx.io.NDArrayIter(data={'data': test_data}, label={'label': mnist['test_label']},
                              batch_size=batch_size)


# used in naive_bayes.py together with alternative way of loading MNIST dataset
# keep it here for comparison
# def transform(data, label):
#     return nd.floor(data / 128).astype('float32').squeeze(axis=-1), label
#
# mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
# mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
# train_images, train_labels = mnist_train[:]


def gpu_exists():
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(0))
    except:
        return False
    return True


def get_model_ctx():
    if gpu_exists():
        print('Using GPU for model_ctx')
        return mx.gpu(0)
    else:
        print('Using CPU for model_ctx')
        return mx.cpu()


model_ctx = get_model_ctx()


def show_mnist():
    mx.random.seed(1)
    print('Test data labels shape : {}'.format(mnist['test_label'].shape))
    print('Train data labels shape : {}'.format(mnist['train_label'].shape))
    n_samples = 10
    idx = np.random.choice(len(mnist['train_data']), n_samples)
    _, axarr = plt.subplots(1, n_samples, figsize=(16, 4))
    for i, j in enumerate(idx):
        axarr[i].imshow(mnist['train_data'][j][0], cmap='Greys')
        axarr[i].get_xaxis().set_ticks([])
        axarr[i].get_yaxis().set_ticks([])
    plt.show()


def set_params():
    n_batches = train_data.shape[0] / batch_size
    print('parameters : n_batches = {}'.format(n_batches))


def main():
    show_mnist()
    set_params()


if __name__ == '__main__':
    npx.set_np()
    main()
    sys.exit(0)
