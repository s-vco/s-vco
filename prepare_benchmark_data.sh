# mmhal
mkdir ./eval_vlm/playground/data/eval/halbench/RLHF-V/images/
jq -r '.[] | "\(.image_src) \(.image_id)"' ./eval_vlm/playground/data/eval/halbench/RLHF-V/eval/data/mmhal-bench_answer_template.json | while read -r url id; do
    wget -O ./eval_vlm/playground/data/eval/halbench/RLHF-V/images/mmhal/${id}.jpg "$url"
done

# mmvet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
mv ./mm-vet/images/ ./eval_vlm/playground/data/eval/mm-vet/images/
rm -r mm-vet mm-vet.zip

# llava-bench-in-the-wild
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild
mv ./llava-bench-in-the-wild/images/ ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/images/
rm -rf llava-bench-in-the-wild

# cvbench
git lfs install
git clone https://huggingface.co/datasets/nyu-visionx/CV-Bench
mv ./CV-Bench/img/ ./eval_vlm/playground/data/eval/cvbench/img/
rm -rf CV-Bench

# mmvp
git lfs install
git clone https://huggingface.co/datasets/MMVP/MMVP
mv "./MMVP/MMVP Images" "./eval_vlm/playground/data/eval/MMVP/MMVP Images"
rm -rf MMVP

# realworldqa
python -c "from datasets import load_dataset; ds = load_dataset('xai-org/RealworldQA'); \
[sample['image'].save(f'eval_vlm/playground/data/eval/realworldqa/images/{i+1}.webp', format='WEBP') for i, sample in enumerate(ds['test'])]"

# textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
mv train_images/ ./eval_vlm/playground/data/eval/textvqa/train_images/
rm -rf train_val_images.zip

# scienceqa
git clone https://github.com/lupantech/ScienceQA.git
cd ScienceQA
mkdir data/scienceqa/images
bash tools/download.sh
cd ..
mv ScienceQA/data/scienceqa/images/ ./eval_vlm/playground/data/eval/scienceqa/images/
rm -rf ScienceQA/
