#!/bin/bash
cd eval_vlm

# export CUDA device variables
export CUDA_VISIBLE_DEVICES="${DEVICE}"

# enable script debugging (optional)
set -x

(

echo "Running evaluation LLAVABENCH for checkpoint ${CKPT_PATH} on CUDA device ${CUDA_VISIBLE_DEVICES}"

# define output file for the generation step
DST_FILE="./playground/data/eval/llava-bench-in-the-wild/answers/${RUN_NAME}.jsonl"

# only run the inference if the DST_FILE does not already exist
if [ ! -f "$DST_FILE" ]; then
    python -m llava.eval.model_vqa_loader \
        --model-base "${MODEL_BASE}" \
        --model-path "${CKPT_PATH}" \
        --fp16 True \
        --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
        --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
        --answers-file "./playground/data/eval/llava-bench-in-the-wild/answers/${RUN_NAME}.jsonl" \
        --temperature 0 \
        --max_new_tokens 1024
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi

# create directory for reviews
mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/llavabench_gpt_review.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/llavabench_gpt_review_rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${RUN_NAME}.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${RUN_NAME}.jsonl

# parse review results
python llava/eval/llavabench_gpt_review_summarize.py \
    -f "playground/data/eval/llava-bench-in-the-wild/reviews/${RUN_NAME}.jsonl"

echo "Finished evaluation for checkpoint path ${RUN_NAME} on device ${DEVICE}"

echo "Writing output to ${OUTPUT_DIR}/llava_bench_in_the_wild.txt"

) > "${OUTPUT_DIR}/llava_bench_in_the_wild.txt" 2>&1 &  # run in background

wait
