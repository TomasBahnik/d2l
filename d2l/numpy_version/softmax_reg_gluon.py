import d2l
from matplotlib import pyplot as plt
from mxnet import gluon, init, npx
from mxnet.gluon import nn

npx.set_np()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 2
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


def predict_ch3(model, t_i, n=6):  # @save
    for X, y in t_i:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(model(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])
    plt.show()


predict_ch3(net, test_iter)
net.save_parameters('softmax.params')