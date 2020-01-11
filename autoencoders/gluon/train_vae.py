import time
import mxnet as mx
from autoencoders.gluon.vae import VAE
from mxnet import gluon, autograd, nd
from tqdm import tqdm, notebook
import autoencoders.gluon.init_vae as init_vae

n_hidden = 400
n_latent = 2
n_layers = 2  # num of dense layers in encoder and decoder respectively
n_output = 784
model_prefix = 'vae_gluon_{}d{}l{}h.params'.format(n_latent, n_layers, n_hidden)
model_ctx = init_vae.model_ctx
train_iter = init_vae.train_iter
test_iter = init_vae.test_iter

net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=init_vae.batch_size)

net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

n_epoch = 50
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []
for epoch in notebook.tqdm(range(n_epoch), desc='epochs'):
    epoch_loss = 0
    epoch_val_loss = 0

    train_iter.reset()
    test_iter.reset()

    n_batch_train = 0
    for batch in train_iter:
        n_batch_train += 1
        data = batch.data[0].as_in_context(model_ctx)
        with autograd.record():
            loss = net(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()

    n_batch_val = 0
    for batch in test_iter:
        n_batch_val += 1
        data = batch.data[0].as_in_context(model_ctx)
        loss = net(data)
        epoch_val_loss += nd.mean(loss).asscalar()

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss)
    validation_loss.append(epoch_val_loss)

    if epoch % max(print_period, 1) == 0:
        tqdm.write('Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))

end = time.time()
print('Time elapsed: {:.2f}s'.format(end - start))
