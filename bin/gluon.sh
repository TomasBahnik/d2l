#!/bin/bash
# Script for creating gluon conda env
#    * gpu 1st argument (via env.sh) means GPU version
SCRIPT=$(realpath "$0")
WORK_DIR=$(dirname "$SCRIPT")
SCRIPT_NAME=$(basename "$0" .sh)

source "$WORK_DIR"/env.sh
# https://github.com/conda/conda/issues/7980
source $CONDA_HOME/etc/profile.d/conda.sh

CONDA_ENV=gluon
PYTHON_VER=3.7
MXNET_VER=1.6.0b20190915

logger "Create Conda env : $CONDA_ENV"
conda create --name "$CONDA_ENV"
logger "Activate Conda env : $CONDA_ENV"
conda activate "$CONDA_ENV"
logger "Install python $PYTHON_VER and pip"
conda install python=$PYTHON_VER pip
logger "Install d2l by pip"
# matplotlib, ipython .. and others
pip install git+https://github.com/d2l-ai/d2l-en
logger "Install mxnet $MXNET_VER by pip"
pip install mxnet==$MXNET_VER
logger "Upgrade mxnet by pip"
pip install -U --pre mxnet
logger "Install tqdm used by VAE"
conda install tqdm
