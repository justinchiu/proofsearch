# https://github.com/wellecks/llemma_formal2formal/blob/master/scripts/eval_llemma7b.sh

MAX_ITERS=100
NUM_SAMPLES=32
TEMPERATURES="0.0"
TIMEOUT=600
NUM_SHARDS=4
DATASET="minif2f-test"
DATA="data/minif2f.jsonl"

MODEL="open-web-math/llemma_7b"
NAME="llemma7b"

OUTPUT_DIR="output/${NAME}_minif2f_test"

#for SHARD in 0 1 2 3
for SHARD in 0
do
  echo HI
  CUDA_VISIBLE_DEVICES=${SHARD} pdm run python -m pdb \
      runsearch.py \
      --dataset-name ${DATASET} \
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
