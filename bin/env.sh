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
# store arguments in a special array
ARGS=("$@")
# get number of elements
ARGS_COUNT=${#ARGS[@]}

# separate directory for all logs
export BASE_LOG_DIR=$PROJECT_DIR/log
mkdir -p "$LOG_DIR"
LOG_FILE=$BASE_LOG_DIR/$SCRIPT_NAME.log

logger "*** New run of script   : $SCRIPT_NAME"
logger "ARGS_COUNT              : $ARGS_COUNT"
logger "PROJECT_DIR             : $PROJECT_DIR"
logger "LOG_DIR                 : $LOG_DIR"
