#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation MMVET for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation/conversion step
DST_FILE="./playground/data/eval/mm-vet/results/${RUN_NAME}.json"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.model_vqa_loader \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file "./playground/data/eval/mm-vet/llava-mm-vet.jsonl" \
        --image-folder "./playground/data/eval/mm-vet/images" \
        --answers-file "./playground/data/eval/mm-vet/answers/${RUN_NAME}.jsonl" \
        --temperature 0 \
        --max_new_tokens 256

    # create directory for results (if not already existing)
    mkdir -p "./playground/data/eval/mm-vet/results"

    python llava/eval/mmvet_convert_for_eval.py \
        --src "./playground/data/eval/mm-vet/answers/${RUN_NAME}.jsonl" \
        --dst "${DST_FILE}"
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

cd playground/data/eval/mm-vet/
mkdir -p "./grades/${RUN_NAME}"

python mm-vet_evaluator.py \
    --gpt_model gpt-4-0613 \
    --num_run 1 \
    --mmvet_path ./ \
    --result_file "./results/${RUN_NAME}.json" \
    --result_path "./grades/${RUN_NAME}"

echo "Finished evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/mmvet.txt"

) > "${OUTPUT_DIR}/mmvet.txt" 2>&1 &  # run in background

wait
