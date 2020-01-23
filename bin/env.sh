function message() {
  if [ "x$2" = "x" ]; then # no second parameter
    echo "$(date '+%d.%m.%Y-%H:%M:%S')" : "$1"
  else
    echo "$(date '+%d.%m.%Y-%H:%M:%S')" : "$1" >>$2
  fi
}

function logger() {
  message "$1" "$LOG_FILE"
}

SCRIPT=$(realpath "$0")
WORK_DIR=$(dirname "$SCRIPT")
SCRIPT_NAME=$(basename "$0" .sh)
PROJECT_DIR=$WORK_DIR/..
DEFAULT_CONDA_ENV=gluon
PYTHON_VER=3.7

MXNET_CPU_VER=1.6.0b20190915
MXNET_GPU_VER=1.6.0b20191122
MXNET_CUDA_VER=cu101

# separate directory for all logs
BASE_LOG_DIR=$PROJECT_DIR/log
mkdir -p "$BASE_LOG_DIR"

# store arguments in a special array
ARGS=("$@")
# get number of elements
ARGS_COUNT=${#ARGS[@]}

# 1. script parameter even if global variable IS defined
if [ "x$1" != "x" ]; then
    export CONDA_ENV=$1
fi


# if env is NOT SET in previous step AND global variable CONDA_ENV IS NOT defined use default
if [ -z "$CONDA_ENV" ]; then
    # log file is not defined yet
    LOG_FILE=$BASE_LOG_DIR/$DEFAULT_CONDA_ENV.log
    message "CONDA_ENV is empty, use default : $DEFAULT_CONDA_ENV"
    export CONDA_ENV=$DEFAULT_CONDA_ENV
fi
LOG_FILE=$BASE_LOG_DIR/$CONDA_ENV.log
CONDA_HOME=~/miniconda3

logger "*** New run of script   : $SCRIPT_NAME"
logger "ARGS_COUNT              : $ARGS_COUNT"
logger "PROJECT_DIR             : $PROJECT_DIR"
logger "BASE_LOG_DIR            : $BASE_LOG_DIR"
logger "CONDA_ENV               : $CONDA_ENV"
logger "CONDA_HOME              : $CONDA_HOME"

# https://github.com/conda/conda/issues/7980
source "$CONDA_HOME"/etc/profile.d/conda.sh

logger "Create Conda env : $CONDA_ENV"
conda create --name "$CONDA_ENV"
logger "Activate Conda env : $CONDA_ENV"
conda activate "$CONDA_ENV"
logger "Install python $PYTHON_VER and pip"
conda install python="$PYTHON_VER" pip