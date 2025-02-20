#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation SCIENCE-QA for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/scienceqa/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.model_vqa_loader \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file "${DST_FILE}" \
        --scienceqa True \
        --single-pred-prompt True \
        --temperature 0 \
        --max_new_tokens 128
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

mkdir -p ./playground/data/eval/scienceqa/outputs/
mkdir -p ./playground/data/eval/scienceqa/results/

python llava/eval/sqa_grader.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file "./playground/data/eval/scienceqa/answers/${RUN_NAME}.jsonl" \
    --output-file "./playground/data/eval/scienceqa/outputs/${RUN_NAME}_output.jsonl" \
    --output-result "./playground/data/eval/scienceqa/results/${RUN_NAME}_result.json"

echo "Finished Science QA evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/scienceqa.txt"

) > "${OUTPUT_DIR}/scienceqa.txt" 2>&1 &  # run in background

wait
