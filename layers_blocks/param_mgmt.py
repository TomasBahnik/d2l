import sys
from mxnet import np, npx
from mxnet.gluon import nn

npx.set_np()


def run_model(input_data):
    net = nn.Sequential()
    net.add(nn.Dense(8, activation='relu'))
    net.add(nn.Dense(1)) # for each input row one output of size 1 (i.e. scalar)
    net.initialize()

    print("net input shape : {}".format(input_data.shape))
    print("net input       : {}".format(input_data))
    net_output = net(input_data)  # Forward computation
    print("net output shape : {}".format(net_output.shape))
    print("net output       : {}".format(net_output))
    print(net[0].params)
    print(net[1].params)
    net_idx=0
    print("bias params for net {}".format(net_idx))
    bias = net[net_idx].bias
    print("param {}".format(bias))
    print("type {}".format(type(bias)))
    print("data {}".format(bias.data()))


def main():
    x = np.random.uniform(size=(3, 8))
    run_model(x)


if __name__ == '__main__':
    main()
    sys.exit(0)
