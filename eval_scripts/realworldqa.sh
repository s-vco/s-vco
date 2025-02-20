#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation REALWORLDQA for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/realworldqa/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.realworldqa_inference \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file ./playground/data/eval/realworldqa/llava_test_realworldqa.jsonl \
        --image-folder ./playground/data/eval/realworldqa/images \
        --answers-file "${DST_FILE}" \
        --temperature 0 \
        --max_new_tokens 128
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

python llava/eval/realworldqa_rule_grader.py \
    --answer_file "./playground/data/eval/realworldqa/answers/${RUN_NAME}.jsonl"

echo "Finished evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/realworldqa.txt"

) > "${OUTPUT_DIR}/realworldqa.txt" 2>&1 &  # run in background

wait
