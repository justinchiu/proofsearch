# https://github.com/wellecks/llemma_formal2formal/blob/master/scripts/eval_llemma7b.sh

NUM_EXAMPLES=100
MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES="0.0"
TIMEOUT=600
NUM_SHARDS=1
DATASET="minif2f-valid"
DATA="data/minif2f.jsonl"

MODEL="open-web-math/llemma_7b"
NAME="llemma7b"

OUTPUT_DIR="output/${NAME}_minif2f_valid"

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
