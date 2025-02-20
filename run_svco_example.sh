# 8-device distributed configs.
NUM_GPUS=8
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

SVCO=True  # identifier for using S-VCO trainer

RUN_NAME=svco_lvint7b_mvc  # experiment name --> used for logging & checkppint-saving directory

MODEL_FAMILY_ID=llava-interleave  # choose from: ["llava-interleave", "llava-1.5"]
MODEL_NAME_OR_PATH="llava-hf/llava-interleave-qwen-7b-hf"  # choose from: ["llava-hf/llava-interleave-qwen-7b-hf", "llava-hf/llava-1.5-7b-hf"] or any other size

FIX_VIT=True  # frozen vision encoder
FIX_MULTIMODAL_PROJECTOR=False  # trainable vl-connector

# beta values in S-VCO objective
BETA_IMG_WIN_VS_NO_IMG_PREFERENCE=0.1
BETA_NO_IMG_VS_IMG_LOSE_PREFERENCE=0.1
BETA_IMG_LOSE_VS_NO_IMG_PREFERENCE=0.1
BETA_NO_IMG_VS_IMG_WIN_PREFERENCE=0.1

# dtype
FP16=True
BF16=False
TF32=True  # in case of occasional FP32 operations

DS_STAGE=zero3  # deepspeed stage "zero3"
USE_FLASH_ATTN=False  # flash_attn enabled

# train_set & validataion_set path
DATASET_PATH=./data/mvc_train.json
EVAL_DATASET_PATH=./data/mvc_val.json

MODEL_MAX_LENGTH=128  # trainable input text max-legnth -- adjustable according to dataset's text length
DATALOADER_NUM_WORKERS=16 

# common training configs
NUM_TRAIN_EPOCHS=1  
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=0.000001
WEIGHT_DECAY=0.02
ADAM_BETA2=0.95
WARMUP_RATIO=0.05
LR_SCHEDULER_TYPE="cosine"

GRADIENT_CHECKPOINTING=True  # save VRAM -- disable this for faster training speed

# eval & logging & saving configs.
EVALUATION_STRATEGY="steps"
EVAL_STEPS=31
SAVE_STRATEGY="steps"
SAVE_STEPS=31  # saving checkpoint at 31-step interval to "./checkpoints/{RUN_NAME}/checkpoint-{STEP}/"
SAVE_TOTAL_LIMIT=100
LOGGING_STEPS=31
LOGGING_FIRST_STEP=True
REPORT_TO=wandb

mkdir -p LOG_TRAIN

nohup torchrun $DISTRIBUTED_ARGS run.py \
    --svco $SVCO \
    --run_name $RUN_NAME \
    --model_family_id $MODEL_FAMILY_ID \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --fix_vit $FIX_VIT \
    --fix_multimodal_projector $FIX_MULTIMODAL_PROJECTOR \
    --beta_img_win_vs_no_img_preference $BETA_IMG_WIN_VS_NO_IMG_PREFERENCE \
    --beta_no_img_vs_img_lose_preference $BETA_NO_IMG_VS_IMG_LOSE_PREFERENCE \
    --beta_img_lose_vs_no_img_preference $BETA_IMG_LOSE_VS_NO_IMG_PREFERENCE \
    --beta_no_img_vs_img_win_preference $BETA_NO_IMG_VS_IMG_WIN_PREFERENCE \
    --fp16 $FP16 \
    --bf16 $BF16 \
    --tf32 $TF32 \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --use_flash_attn $USE_FLASH_ATTN \
    --dataset_path $DATASET_PATH \
    --eval_dataset_path $EVAL_DATASET_PATH \
    --model_max_length $MODEL_MAX_LENGTH \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta2 $ADAM_BETA2 \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --logging_first_step $LOGGING_FIRST_STEP \
    --report_to $REPORT_TO \
    --output_dir ./checkpoints/${RUN_NAME}/ \
    > LOG_TRAIN/${RUN_NAME}.txt 2>&1 &
