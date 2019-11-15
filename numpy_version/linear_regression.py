import sys
import d2l
from mxnet import autograd, np, npx
import random
from matplotlib import pyplot as plt

npx.set_np()

BATCH_SIZE = 10
NUM_EPOCHS = 4  # Number of iterations
LR = 0.03  # Learning rate

measurement_count = 2000
w1 = 2
w2 = -3.4
w3 = 6.7
true_w = np.array([w1, w2, w3])  # shape or true_ must be the same as shape of  w
true_b = 4.2  # shape (1,)
# initial values of weights and bias with the same shape as true ones
w = np.random.normal(0, 0.01, (3, 1))
b = np.zeros(1)
# attach grad attribute to all parameters
w.attach_grad()
b.attach_grad()

net = d2l.linreg  # Our fancy linear model
loss = d2l.squared_loss  # 0.5 (y-y')^2


# generator function
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def train_model(num_epochs, batch_size, lr):
    features, labels = d2l.synthetic_data(true_w, true_b, measurement_count)
    for epoch in range(num_epochs):
        # Assuming the number of examples can be divided by the batch size, all
        # the examples in the training dataset are used once in one epoch
        # iteration. The features and tags of minibatch examples are given by X
        # and y respectively
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)  # Minibatch loss in X and y
            l.backward()  # Compute gradient on l with respect to [w,b]
            d2l.sgd([w, b], lr, batch_size)  # Update parameters using their gradient
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

    rw = w.reshape(true_w.shape)
    aew = true_w - rw
    rew = aew / rw * 100
    aeb = true_b - b
    reb = aeb / b * 100
    print('estimated w = {}, true w = {}'.format(rw, true_w))
    print('estimated b = {}, true b = {}'.format(b, true_b))
    print('Abs error in estimating w', aew)
    print('Abs error in estimating b', aeb)
    print('Relative error of w : {} %'.format(rew))
    print('Relative error of b : {} %'.format(reb))
    # 4,2001953


def show_data():
    features, labels = d2l.synthetic_data(true_w, true_b, measurement_count)
    print("feature size {}, label size {}".format(features.size, labels.size))
    print("feature shape {}, label shape {}".format(features.shape, labels.shape))
    print('features:', features[0], '\nlabel:', labels[0])
    d2l.set_figsize((3.5, 2.5))
    d2l.plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    d2l.plt.scatter(features[:, 0].asnumpy(), labels.asnumpy(), 1)
    plt.show()


def show_batches(batch_size):
    features, labels = d2l.synthetic_data(true_w, true_b, measurement_count)
    for X, y in data_iter(batch_size, features, labels):
        print("features batch shape {}, labels batch shape {}".format(X.shape, y.shape))
        print("features batch size {}, labels batch size {}".format(X.size, y.size))
        print("{}.\n ====\n features batch====\n {}, \n labels batch====\n {}".format(k, X, y))
        # remove this `break`  to see all data
        break


def main():
    # show_data()
    # show_batches(BATCH_SIZE)
    train_model(NUM_EPOCHS, BATCH_SIZE, LR)


if __name__ == '__main__':
    main()
    sys.exit(0)
