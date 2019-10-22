import d2l
import math
from mxnet import nd, gluon
from matplotlib import pyplot as plt

d2l.use_svg_display()


def transform(data, label):
    return nd.floor(data / 128).astype('float32').squeeze(axis=-1), label


mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)


def main():
    image, label = mnist_train[3]
    print(image.shape, label, type(image))
    print(label, type(label), label.dtype)
    images, labels = mnist_train[10:38]
    print(images.shape, labels.shape)
    d2l.show_images(images, 2, 9)
    plt.show()


if __name__ == '__main__':
    main()
