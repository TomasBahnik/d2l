import d2l
import math
from mxnet import nd, gluon
from matplotlib import pyplot as plt

d2l.use_svg_display()


def transform(data, label):
    return nd.floor(data / 128).astype('float32').squeeze(axis=-1), label


mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

number_of_classes = 10  # 0 .. 9
image_size = 28
train_images, train_labels = mnist_train[:]  # all training examples


def main():
    train_model(train_images, train_labels)


def train_model(images, labels):
    l_c = label_counts(labels)
    label_probabilities = l_c / l_c.sum()
    for label, probability in enumerate(label_probabilities):
        print('label {} : {}'.format(label, probability))
    print('Samples', nd.size_array(images))
    P_xy = label_pixel_probabilities(images, labels, l_c)
    # test the prediction
    image, label = mnist_test[0]
    prediction = bayes_pred(image, label_probabilities, P_xy)
    print(prediction)


def label_counts(labels):
    """Returns a new array where element at index = `label` contains the count of the `label`
     in `labels`

        Parameters
        ----------
        labels : int type labels
            The 1D arrays of int valued labels in [0,number_of_classes)
        """

    ret_val = nd.zeros(number_of_classes)
    for label in range(number_of_classes):
        ret_val[label] = (labels == label).sum()
    return ret_val


def label_pixel_probabilities(images, labels, lbl_counts):
    label_pixel_counts = nd.zeros((number_of_classes, image_size, image_size))
    for label in range(number_of_classes):
        # images for given label
        label_images = images.asnumpy()[labels == label]
        # numpy arrays : element wise sum = total counts of pixels for given label because pixel is 0 or 1
        label_pixel_counts_numpy = label_images.sum(axis=0)
        label_pixel_counts[label] = nd.array(label_pixel_counts_numpy)
    # Laplace smoothing, add 1
    label_counts_smoothed_reshaped = (lbl_counts + 1).reshape((10, 1, 1))
    P_xy = (label_pixel_counts + 1) / label_counts_smoothed_reshaped
    d2l.show_images(P_xy, 5, 2)  # 2x5=10
    plt.show()
    return P_xy


def bayes_pred(x, P_y, P_xy):
    x = x.expand_dims(axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy) * (1 - x)
    p_xy = p_xy.reshape((10, -1)).prod(axis=1)  # p(x|y)
    return p_xy * P_y


def show_samples():
    image, label = mnist_train[2]
    print(image.shape, label, type(image))
    print(label, type(label), label.dtype)
    images, labels = mnist_train[0:4:1]
    print(images.shape, labels.shape)
    d2l.show_images(images, 2, 9)
    plt.show()


if __name__ == '__main__':
    main()
