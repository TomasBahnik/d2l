import d2l
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


def show_samples():
    image, label = mnist_train[2]
    print(image.shape, label, type(image))
    print(label, type(label), label.dtype)
    images, labels = mnist_train[0:4:1]
    print(images.shape, labels.shape)
    d2l.show_images(images, 2, 9)
    plt.show()


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
    # show pixel probabilities
    # d2l.show_images(P_xy, 5, 2)  # 2x5=10
    # plt.show()
    return P_xy


# multiplication of small probabilities leads to zeroes
def bayes_pred_underflow(x, P_y, P_xy):
    x = x.expand_dims(axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy) * (1 - x)
    p_xy = p_xy.reshape((10, -1)).prod(axis=1)  # p(x|y)
    return p_xy * P_y


# logarithm of probabilities product


def bayes_pred_stable(x, P_y, P_xy):
    log_P_xy = nd.log(P_xy)
    log_P_xy_neg = nd.log(1 - P_xy)
    log_P_y = nd.log(P_y)
    x = x.expand_dims(axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape((10, -1)).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y


# verify prediction for one image
def predict_one(x, y, P_y, P_xy):
    # label probability predictions
    py = bayes_pred_stable(x, P_y, P_xy)
    print(py)
    # Check if the label prediction is correct
    predicted_label = int(py.argmax(axis=0).asscalar())
    print('predicted label {} : actual label {} : match {}'.format(predicted_label, y, predicted_label == y))


# prediction for set of images
def predict(X, P_y, P_xy):
    return [int(bayes_pred_stable(x, P_y, P_xy).argmax(axis=0).asscalar()) for x in X]


def accuracy(P_y, P_xy):
    X, y = mnist_test[:]
    py = predict(X, P_y, P_xy)
    matches = (nd.array(py).asnumpy() == y).sum()
    print('matches : {}, total labels : {} => accuracy = {}'.format(matches, len(y), matches / len(y)))
    return matches / len(y)


def train_model(images, labels):
    l_c = label_counts(labels)
    # label probabilities
    P_y = l_c / l_c.sum()
    for label, probability in enumerate(P_y):
        print('label {} : {}'.format(label, probability))
    print('Samples', nd.size_array(images))
    P_xy = label_pixel_probabilities(images, labels, l_c)
    # test the prediction
    image, label = mnist_test[1]
    predict_one(image, label, P_y, P_xy)
    X, y = mnist_test[:18]
    d2l.show_images(X, 2, 9, titles=predict(X, P_y, P_xy))
    plt.show()
    ac = accuracy(P_y, P_xy)
    print('accuracy  : {}'.format(ac))


def main():
    train_model(train_images, train_labels)


if __name__ == '__main__':
    main()
