#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation MMVP for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/MMVP/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.mmvp_inference \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --directory ./playground/data/eval/MMVP \
        --answers-file "${DST_FILE}" \
        --temperature 0 \
        --max_new_tokens 128
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

python llava/eval/mmvp_rule_grader.py \
    --answer_file "./playground/data/eval/MMVP/answers/${RUN_NAME}.jsonl"

echo "Finished evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/mmvp.txt"

) > "${OUTPUT_DIR}/mmvp.txt" 2>&1 &  # run in background

wait
