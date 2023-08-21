PARTITION=$1
JOB_NAME=$2
PRED_DIR=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}
PYTHONPATH=. srun -p ${PARTITION} --job-name=${JOB_NAME} -n1 --cpus-per-task=5  ${SRUN_ARGS} python video_inference/eval_video.py ${PRED_DIR} ${PY_ARGS}
