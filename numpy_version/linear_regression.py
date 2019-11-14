import sys
import d2l
from mxnet import autograd, np, npx
import random
from matplotlib import pyplot as plt

npx.set_np()

number_of_samples = 1000
w1 = 2
w2 = -3.4
true_w = np.array([w1, w2])
true_b = 4.2


def show_data():
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    print("feature size {}, label size {}".format(features.size, labels.size))
    print("feature shape {}, label shape {}".format(features.shape, labels.shape))
    print('features:', features[0], '\nlabel:', labels[0])
    d2l.set_figsize((3.5, 2.5))
    d2l.plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    d2l.plt.scatter(features[:, 0].asnumpy(), labels.asnumpy(), 1)
    plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def show_batches(batch_size):
    features, labels = d2l.synthetic_data(true_w, true_b, number_of_samples)
    for X, y in data_iter(batch_size, features, labels):
        print("features batch shape {}, labels batch shape {}".format(X.shape, y.shape))
        print("features batch size {}, labels batch size {}".format(X.size, y.size))
        print("features batch {}, \n labels batch {}".format(X, y))
        break


def main():
    # show_data()
    show_batches(10)


if __name__ == '__main__':
    main()
    sys.exit(0)
