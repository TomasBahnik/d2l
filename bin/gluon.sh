#!/bin/bash
# Script for creating gluon conda env
#    * env name  is teh 1st argument (via env.sh). If it contains gpu
#    GPU version is installed
SCRIPT=$(realpath "$0")
WORK_DIR=$(dirname "$SCRIPT")

# https://github.com/koalaman/shellcheck/wiki/SC1090
# shellcheck source=env.sh
source "$WORK_DIR"/env.sh

# matplotlib, ipython .. and others for d2l.ai
logger "Install d2l by pip"
pip install git+https://github.com/d2l-ai/d2l-en
if [[ "$CONDA_ENV" == *"gpu"* ]]; then
  logger "Install MXNET GPU version '$MXNET_CUDA_VER'"
  pip install mxnet-$MXNET_CUDA_VER=="$MXNET_GPU_VER"
  logger "Upgrade mxnet by pip"
  pip install -U --pre mxnet-$MXNET_CUDA_VER
else
  logger "Install MXNET CPU version $MXNET_CPU_VER"
  pip install mxnet==$MXNET_CPU_VER
  logger "Upgrade mxnet by pip"
  pip install -U --pre mxnet
fi
logger "Install tqdm used by VAE"
conda install tqdm
