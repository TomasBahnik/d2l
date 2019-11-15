import sys
import d2l
import common
from mxnet import autograd, np, npx, gluon
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss

npx.set_np()

BATCH_SIZE = 10
NUM_EPOCHS = 3  # Number of iterations
LR = 0.03  # Learning rate
MEASUREMENT_COUNT = 2000  # number of examples

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, MEASUREMENT_COUNT)

# Sequential defines a container for several layers that will be chained together.
net = nn.Sequential()

# when we execute net(X) later, Gluon will automatically infer the number of inputs to each layer.
net.add(nn.Dense(1))

# each weight parameter should be randomly sampled from a normal distribution with mean  0
# and standard deviation  0.01
# The bias parameter will be initialized to zero by default.
# Both the weight vector and bias will have attached gradients
# Unknown dimensions of parameter vector.
# Initialization is actually deferred. The real initialization will take place only
# when we for the first time attempt to pass data through the network.
net.initialize(init.Normal(sigma=0.01))

# The squared loss is also known as the L2 norm loss but l2 norm is ||x||_2 = (sum |x_i|^2)^1/2
loss = gloss.L2Loss()

# Gluon ArrayDataset and DataLoader
data_iter = d2l.load_array((features, labels), BATCH_SIZE)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': LR})


def train_model_gluon(num_epochs, batch_size):
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                ls = loss(net(X), y)
            ls.backward()
            trainer.step(batch_size)
        ls = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, ls.mean().asnumpy()))

    w = net[0].weight.data()
    b = net[0].bias.data()
    common.model_errors(w, true_w, b, true_b)


def main():
    train_model_gluon(NUM_EPOCHS, BATCH_SIZE)


if __name__ == '__main__':
    main()
    sys.exit(0)
