import d2l
from matplotlib import pyplot as plt
from mxnet import np, npx

npx.set_np()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(32)
for X, y in train_iter:
    print(X.shape)
    break

num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()


def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcast mechanism is applied here


X = np.random.normal(size=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)


def net(X):
    return softmax(np.dot(X.reshape(-1, num_inputs), W) + b)


def cross_entropy(y_hat, y):
    return - np.log(y_hat[range(len(y_hat)), y])


accuracy = d2l.evaluate_accuracy(net, test_iter)
print("accuracy of the untrained model : {}".format(accuracy))

num_epochs, lr = 2, 0.1


def updater(b_size):
    return d2l.sgd([W, b], lr, b_size)


d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  # @save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])
    plt.show()


predict_ch3(net, test_iter)
