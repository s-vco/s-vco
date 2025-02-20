#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation TEXT-VQA for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/textvqa/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.model_vqa_loader \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file "${DST_FILE}" \
        --temperature 0 \
        --max_new_tokens 128
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

python -m llava.eval.textvqa_grader \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${DST_FILE}"

echo "Finished TextVQA evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/textvqa.txt"

) > "${OUTPUT_DIR}/textvqa.txt" 2>&1 &  # run in background

wait
