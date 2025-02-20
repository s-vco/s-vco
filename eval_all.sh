# parse arguments into variables (strings)
while [ $# -gt 0 ]; do
  case "$1" in
    --model_base=*)
      MODEL_BASE="${1#*=}"
      ;;
    --run_name=*)
      RUN_NAME="${1#*=}" 
      ;;
    --ckpt_path=*)
      CKPT_PATH="${1#*=}" 
      ;;
    --openai_key=*)
      OPENAI_API_KEY="${1#*=}"
      ;;
    --device=*)
      DEVICE="${1#*=}"
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

# export arguments as strings
export MODEL_BASE
export RUN_NAME
export CKPT_PATH
export OPENAI_API_KEY
export DEVICE
export OUTPUT_DIR="$(pwd)/LOG_EVAL/${RUN_NAME}"

# debug prints (optional)
echo "MODEL_BASE = ${MODEL_BASE}"  # ["llava-hf/llava-interleave-qwen-7b-hf", "llava-hf/llava-1.5-7b-hf"]
echo "RUN_NAME = ${RUN_NAME}"  # experiment name --> used for eval result logging
echo "CKPT_PATH = ${CKPT_PATH}" # ABSOLUTE path of checkpoints
echo "OPENAI_API_KEY = ${OPENAI_API_KEY}"  # needed for mmhal, mmvet, llavabench
echo "DEVICE = CUDA:${DEVICE}"  # cuda device id
echo "OUTPUT_DIR = ${OUTPUT_DIR}"  # cuda device id

mkdir -p ${OUTPUT_DIR}

# run each benchmark in sequence: each benchmark has a corresponding script under 'eval_scripts/{benchmark_name}.sh'
echo "========== Evaluating mmhal =========="
bash eval_scripts/mmhal.sh

echo "========== Evaluating mmvet =========="
bash eval_scripts/mmvet.sh

echo "========== Evaluating llava_bench_in_the_wild =========="
bash eval_scripts/llavabench.sh

echo "========== Evaluating cvbench =========="
bash eval_scripts/cvbench.sh

echo "========== Evaluating mmvp =========="
bash eval_scripts/mmvp.sh

echo "========== Evaluating realworldqa =========="
bash eval_scripts/realworldqa.sh

echo "========== Evaluating textvqa =========="
bash eval_scripts/textvqa.sh

echo "========== Evaluating scienceqa =========="
bash eval_scripts/sqa.sh

echo "Benchmark Eval All Done --> Eval logs saved to LOG_EVAL/${CKPT_PATH}/..."
