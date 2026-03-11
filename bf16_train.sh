set -ex
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

lines=`echo $POD_NAME | grep edljob | wc -l`
if [ $lines -eq 0 ]; then
  WORLD_SIZE=${WORLD_SIZE:-1}
  NODE_RANK=${RANK:-0}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  RANDOM_PORT=$[$RANDOM + 20000]
  MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
  GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
  echo "---> from pytorch runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
  LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${WORLD_SIZE} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "
else
  WORLD_SIZE=${WORKER_NUM:-1}
  NODE_RANK=${RANK:-0}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  RANDOM_PORT=$[$RANDOM + 20000]
  MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
  GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
  echo "---> from edl runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}"
  LAUNCHER=" \
       python -m torch.distributed.run \
       --nnode=$WORLD_SIZE \
       --nproc_per_node=$GPUS_PER_NODE \
       "
fi

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG_SUBSYS=INIT   # disable aistudio default nccl env

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NVTE_DEBUG=0
pip install blobfile tiktoken


DEVICE_MODEL=$(nvidia-smi -i 0 -q | grep "Product Name" | awk -F: '{ print $2 }')
DEVICE_MODEL=$(echo "$DEVICE_MODEL" | xargs)  # drop white space

if [[ $DEVICE_MODEL == NVIDIA* ]]; then
    DEVICE_MODEL=${DEVICE_MODEL#"NVIDIA"}
    DEVICE_MODEL=$(echo "$DEVICE_MODEL" | sed 's/^ *//')
fi


JOB_DIR="/tmp/input/atorch/logs/ling_bf16_test"
OLD_JOB_DIR="/input/atorch/logs/ling_bf16_new"
mkdir -p ${JOB_DIR}
CHECKPOINT_PATH=${OLD_JOB_DIR} #<Specify path>
TENSORBOARD_LOGS_PATH="/tmp/logs/tfevent"


LOG_PATH="${JOB_DIR}/log_${NODE_RANK}.txt"

MOE_ARGS=(
    --untie-embeddings-and-output-weights
    --swiglu
    --use-mcore-models
    --transformer-impl transformer_engine
    --disable-bias-linear
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --attention-backend auto
    --hidden-dropout 0
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 5120
    --qk-layernorm
    --max-position-embeddings 4096
    --attention-dropout 0
    --num-experts 256
    --moe-layer-freq [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type aux_loss
    --moe-z-loss-coeff 0.0000035
    --moe-router-topk 8
    --moe-router-topk-scaling-factor 2.5
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall

    --moe-shared-expert-overlap
    --moe-permute-fusion 
    --moe-aux-loss-coeff 0.001

)


GBS=768
MBS=1
TP_SIZE=1
PP_SIZE=1
CP_SIZE=1
EP_SIZE=8
SEED=42

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --seq-length "4096"
    --train-iters 1000000
    --lr "0.000374"
    --weight-decay 0.1
    --lr-decay-style constant
    --min-lr "0.000374"
    --lr-warmup-iters 2000
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --init-method-std 0.02
    --bf16
    --seed ${SEED}
)

MODEL_PARALLEL_ARGS=(
    --expert-model-parallel-size ${EP_SIZE}
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-size ${PP_SIZE}
    --tensor-model-parallel-size ${TP_SIZE}
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
)


DATA_DIR="/input/atorch/nv_train_data/"
DATA_PATH="${DATA_DIR}/algebraic_stack_text_document ${DATA_DIR}/pes2o_text_document ${DATA_DIR}/arxiv_text_document \
${DATA_DIR}/open-web-math_text_document ${DATA_DIR}/shard_01_text_document ${DATA_DIR}/shard_02_text_document \
${DATA_DIR}/shard_03_text_document ${DATA_DIR}/shard_04_text_document ${DATA_DIR}/shard_05_text_document \
${DATA_DIR}/shard_06_text_document ${DATA_DIR}/shard_07_text_document ${DATA_DIR}/shard_08_text_document \
${DATA_DIR}/shard_09_text_document ${DATA_DIR}/shard_10_text_document ${DATA_DIR}/starcoder_text_document \
${DATA_DIR}/wiki_text_document"


TOKENIZER_MODEL="/input/atorch/moonlight_config"



DATA_ARGS=(
    --data-path ${DATA_PATH}
    --trust-remote-code
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 949,50,1
    --num-workers 8
)

SAMPLES=135235850
EVAL_AND_LOGGING_ARGS=(
    --save-interval 150000000
    --eval-interval 300000000
    --save /tmp$CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --no-save-optim
    --no-save-rng
    --no-load-optim
    --no-load-rng
    --ckpt-format "torch_dist"
    --eval-iters 1
    --log-interval 1
    --log-throughput
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-validation-ppl-to-tensorboard
)

KERNEL_ARGS=(
    --attention-backend auto
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32 
    --cross-entropy-loss-fusion
)

PROFILING_ARGS=(
    #--profile
    --use-pytorch-profiler
    --profile-ranks 0 1 2 3 4 5 6 7
    --profile-step-start 3
    --profile-step-end 4
)

CMD="${LAUNCHER} pretrain_gpt.py \
    ${MOE_ARGS[@]}
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${KERNEL_ARGS[@]} \
    ${PROFILING_ARGS[@]}"

echo ${CMD}
NCCL_DEBUG=error ${CMD} 2>&1 | tee ${LOG_PATH}


