### Executable
```text
#  chmod +x ./bin/*.sh
#  git update-index --chmod=+x ./bin/*.sh
```
## Ubuntu
   * wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   * chmod 755 Miniconda3-latest-Linux-x86_64.sh
   * bash Miniconda3-latest-Linux-x86_64.sh
   * source ~/.bashrc : Execute commands from a file in the current shell
   * [gluon install](../bin/gluon.sh)
   * [keras install](../bin/keras.sh)
   * conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
     from https://pytorch.org/,stable,cuda 10.1 (only),
   * conda list PIL, if `7.x` conda install pillow==6.1
     (pytorch impl of VAE : PIL `7.x` causes error `cannot import name 'PILLOW_VERSION' from 'PIL'`)   
   * mkdir git
   * cd git/
   * git clone https://github.com/TomasBahnik/d2l
   * python <file>.py

### MXNET GPU
   * pip uninstall mxnet (remove CPU version first) 
   * sudo apt install nvidia-cuda-toolkit (for `nvcc` > 2GB disk space)
   * pip install mxnet-cu101==1.6.0b20191122
   * pip install -U --pre mxnet-cu101 (nothing newer)
   * VAE ~ 364MiB
   
### Keras GPU
   * pip uninstall keras
   * pip uninstall tensorflow
   * conda install -c anaconda keras-gpu (conda install -c anaconda tensorflow-gpu ?)

```text
The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _tflow_select-2.1.0        |              gpu           2 KB  anaconda
    ca-certificates-2019.11.27 |                0         132 KB  anaconda
    certifi-2019.11.28         |           py37_0         156 KB  anaconda
    cudnn-7.6.5                |       cuda10.1_0       250.6 MB  anaconda
    cupti-10.1.168             |                0         1.7 MB  anaconda
    keras-gpu-2.2.4            |                0           5 KB  anaconda
    openssl-1.1.1              |       h7b6447c_0         5.0 MB  anaconda
    tensorboard-1.14.0         |   py37hf484d3e_0         3.2 MB  anaconda
    tensorflow-1.14.0          |gpu_py37h74c33d7_0           4 KB  anaconda
    tensorflow-base-1.14.0     |gpu_py37he45bfe2_0       327.9 MB  anaconda
    tensorflow-estimator-1.14.0|             py_0         291 KB  anaconda
    tensorflow-gpu-1.14.0      |       h0d30ee6_0           3 KB  anaconda
    ------------------------------------------------------------
                                           Total:       589.0 MB

The following NEW packages will be INSTALLED:

  cudnn              anaconda/linux-64::cudnn-7.6.5-cuda10.1_0
  cupti              anaconda/linux-64::cupti-10.1.168-0
  keras-gpu          anaconda/linux-64::keras-gpu-2.2.4-0
  tensorflow-gpu     anaconda/linux-64::tensorflow-gpu-1.14.0-h0d30ee6_0

The following packages will be UPDATED:

  openssl              pkgs/main::openssl-1.1.1d-h7b6447c_3 --> anaconda::openssl-1.1.1-h7b6447c_0

The following packages will be SUPERSEDED by a higher-priority channel:

  _tflow_select          pkgs/main::_tflow_select-2.3.0-mkl --> anaconda::_tflow_select-2.1.0-gpu
  ca-certificates                                 pkgs/main --> anaconda
  certifi                                         pkgs/main --> anaconda
  tensorboard        pkgs/main/noarch::tensorboard-1.15.0-~ --> anaconda/linux-64::tensorboard-1.14.0-py37hf484d3e_0
  tensorflow         pkgs/main::tensorflow-1.15.0-mkl_py37~ --> anaconda::tensorflow-1.14.0-gpu_py37h74c33d7_0
  tensorflow-base    pkgs/main::tensorflow-base-1.15.0-mkl~ --> anaconda::tensorflow-base-1.14.0-gpu_py37he45bfe2_0
  tensorflow-estima~ pkgs/main::tensorflow-estimator-1.15.~ --> anaconda::tensorflow-estimator-1.14.0-py_0

```