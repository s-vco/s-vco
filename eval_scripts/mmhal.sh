#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation MMHAL for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/halbench/RLHF-V/mmhal/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.model_vqa_loader \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file ./playground/data/eval/halbench/RLHF-V/eval/data/mmhal-bench_with_image_eval.jsonl \
        --image-folder ./playground/data/eval/halbench/RLHF-V/images/mmhal \
        --answers-file "${DST_FILE}" \
        --temperature 0 \
        --max_new_tokens 1024
else
    echo "File ${DST_FILE} already exists. Skipping generation step."
fi

# run the evaluation step
cd ./playground/data/eval/halbench/RLHF-V/

mkdir -p ./mmhal/templates/

# convert answers into templates
python eval/change_mmhal_predict_template.py \
    --response-template ./eval/data/mmhal-bench_answer_template.json \
    --answers-file ./mmhal/answers/${RUN_NAME}.jsonl \
    --save-file ./mmhal/templates/${RUN_NAME}.json

mkdir -p ./mmhal/eval_results/

# evaluate using GPT (gpt-4-turbo in this example)
python eval/eval_gpt_mmhal.py \
    --gpt-model gpt-4-turbo \
    --response ./mmhal/templates/${RUN_NAME}.json \
    --evaluation ./mmhal/eval_results/${RUN_NAME}.json \
    --api-key "${OPENAI_API_KEY}"

# summarize metrics
python eval/summarize_gpt_mmhal_review.py \
    ./mmhal/eval_results/${RUN_NAME}.json

echo "Finished mmhal evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/mmhal.txt"

) > "${OUTPUT_DIR}/mmhal.txt" 2>&1 &  # run in background

wait

