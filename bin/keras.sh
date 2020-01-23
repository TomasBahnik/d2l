#!/bin/bash
# Script for creating gluon conda env
#  * env name  is teh 1st argument (via env.sh). If it contains gpu
#    GPU version is installed
SCRIPT=$(realpath "$0")
WORK_DIR=$(dirname "$SCRIPT")

# https://github.com/koalaman/shellcheck/wiki/SC1090
# shellcheck source=env.sh
source "$WORK_DIR"/env.sh

conda install keras
if [[ "$CONDA_ENV" == *"gpu"* ]]; then
  logger "Install Tensorflow GPU"
  conda install -c anaconda tensorflow-gpu
  # ???
  #logger "Install Keras GPU"
  #conda install -c anaconda keras-gpu
fi
logger "Install matplotlib used by VAE"
conda install matplotlib
logger "Install pydot used by VAE"
conda install pydot
