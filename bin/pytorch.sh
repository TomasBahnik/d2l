#!/bin/bash
# Script for creating gluon conda env
#  * env name  is teh 1st argument (via env.sh). If it contains gpu
#    GPU version is installed
SCRIPT=$(realpath "$0")
WORK_DIR=$(dirname "$SCRIPT")

# https://github.com/koalaman/shellcheck/wiki/SC1090
# shellcheck source=env.sh
source "$WORK_DIR"/env.sh

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
if [[ "$CONDA_ENV" == *"gpu"* ]]; then
  logger "Install pytorch GPU"
  conda install -c anaconda pytorch-gpu
fi
logger "List installed pillow"
conda list PIL
# pytorch impl of VAE : PIL `7.x` causes error `cannot import name 'PILLOW_VERSION' from 'PIL'
logger "Install pillow 6.1 used by VAE"
conda install pillow==6.1
logger "mkdir results for reconstruction images"
mkdir -p "$VAE_DIR/pytorch/results"

