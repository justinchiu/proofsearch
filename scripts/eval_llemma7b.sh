# https://github.com/wellecks/llemma_formal2formal/blob/master/scripts/eval_llemma7b.sh

NUM_EXAMPLES=100
MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES="0.0"
TIMEOUT=600
NUM_SHARDS=1

#DATASET="minif2f-valid"
#OUTPUT_DIR="output/${NAME}_minif2f_valid"
DATASET="minif2f-test"
OUTPUT_DIR="output/${NAME}_minif2f_test"

DATA="data/minif2f.jsonl"

MODEL="EleutherAI/llemma_7b"
NAME="llemma7b"

#for SHARD in 0 1 2 3
for SHARD in 0
do
  CUDA_VISIBLE_DEVICES=${SHARD} pdm run python \
      runsearch.py \
      --dataset-name ${DATASET} \
      --num-examples ${NUM_EXAMPLES} \
      --temperatures ${TEMPERATURES} \
      --timeout ${TIMEOUT} \
      --num-shards ${NUM_SHARDS} \
      --shard ${SHARD} \
      --model-name ${MODEL} \
      --max-iters ${MAX_ITERS} \
      --dataset-path ${DATA} \
      --num-samples ${NUM_SAMPLES} \
      --early-stop \
      --output-dir ${OUTPUT_DIR} #\
      #&> ${NAME}_shard${SHARD}.out &

done
