import sys
from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()


class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.output = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def forward(self, x):
        return self.output(self.hidden(x))


def run_model(input_data):
    net = MLP()
    net.initialize()
    print("net input shape : {}".format(input_data.shape))
    print("net input       : {}".format(input_data))
    net_output = net(input_data)
    print("net output shape : {}".format(net_output.shape))
    print("net output       : {}".format(net_output))


def main():
    x = np.random.uniform(size=(2, 20))
    run_model(x)


if __name__ == '__main__':
    main()
    sys.exit(0)
