import d2l
from matplotlib import pyplot as plt
from mxnet import npx, gluon

npx.set_np()

d2l.use_svg_display()

mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)

print("train length : {}, test length: {}".format(len(mnist_train), len(mnist_test)))

X, y = mnist_train[:18]
# d2l.show_images(X.squeeze(axis=-1), 2, 9, titles=d2l.get_fashion_mnist_labels(y))
# plt.show()

batch_size = 256
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=d2l.get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print("loading dada takes {:.2f} sec".format(timer.stop()))

train_iter, test_iter = d2l.load_data_fashion_mnist(32, (64, 64))
for X, y in train_iter:
    print(X.shape)
    break
