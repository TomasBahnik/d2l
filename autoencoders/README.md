Implementation of the [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) in 3 DL frameworks

   * [MXNET/gluon VAE](https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html)
   * [Keras VAE](https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py)
   * [Pytorch VAE](https://github.com/pytorch/examples/tree/master/vae)

### Prepare envs
Conda envs are created by corresponding scripts in [bin folder](../bin). Env name is the 1st argument. If the name of 
environment contains `gpu` GPU version is installed e.g. `./gluon.sh gluon-gpu` creates conda env with MXNET/gluon GPU.

### Run   
Run `python vae.py` in corresponding folder.

### Comparison
GPU usage measured by 

```text
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

and
```text
sudo watch nvidia-smi
```

[GPU consumption for different DL tools](../docs/vae)  
 