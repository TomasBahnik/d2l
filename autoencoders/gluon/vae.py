# from https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html
from mxnet import gluon
from mxnet.gluon import nn

import autoencoders.gluon.init_vae as init_vae


class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, batch_size=100, act_type='relu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        # note to self: requring batch_size in model definition is sad, not sure how to deal with this otherwise though
        super(VAE, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')
            for i in range(n_layers):
                self.encoder.add(nn.Dense(n_hidden, activation=act_type))
            self.encoder.add(nn.Dense(n_latent * 2, activation=None))

            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        h = self.encoder(x)
        # print(h)
        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu
        # eps = F.random_normal(loc=0, scale=1, shape=mu.shape, ctx=model_ctx)
        # this would work fine only for nd (i.e. non-hybridized block)
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=init_vae.get_model_ctx())
        z = mu + F.exp(0.5 * lv) * eps
        y = self.decoder(z)
        self.output = y

        KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
        logloss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
        loss = -logloss - KL

        return loss
