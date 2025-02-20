import argparse
import torch
import os
import json
import math
from tqdm import tqdm
import shortuuid
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pandas as pd

from chat_template_monkey_patch import apply_chat_template


def disable_torch_init():
    """disable the redundant torch default initialization to accelerate model creation"""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def split_list(lst, n):
    """split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_llava(model_path, model_base, dtype="fp32"):
    if dtype=="fp16":
        if "llava-interleave" in model_base:
            model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            processor = AutoProcessor.from_pretrained(model_base, vision_feature_select_strategy="default")
        elif "llava-1.5" in model_base:
            model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            processor = AutoProcessor.from_pretrained(model_base, add_eos_token=True)
            processor.tokenizer.apply_chat_template = apply_chat_template.__get__(processor.tokenizer)  # align template with traning
    elif dtype=="fp32":
        if "llava-interleave" in model_base:
            model = LlavaForConditionalGeneration.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_base, vision_feature_select_strategy="default")
        elif "llava-1.5" in model_base:
            model = LlavaForConditionalGeneration.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_base, add_eos_token=True)
            processor.tokenizer.apply_chat_template = apply_chat_template.__get__(processor.tokenizer)  # align template with traning
    return model, processor

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model, processor = load_llava(model_path, args.model_base, dtype="fp16" if args.fp16 else "fp32")
    model.cuda()
    
    benchmark_dir = os.path.join(args.directory, 'Questions.csv')
    df = pd.read_csv(benchmark_dir)  # assuming the fields are separated by tabs
    
    answers_file = os.path.expanduser(args.answers_file)    
    if os.path.dirname(answers_file):
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            photo_id = index+1
            cur_prompt = row['Question'] + " " + row['Options'] + "\n" + "Answer with the option's letter from the given choices directly."
            qs = cur_prompt
            image_path = os.path.join(args.directory, 'MMVP Images', f"{photo_id}.jpg")
            image = Image.open(image_path).convert('RGB')
            
            if args.no_vision_input:
                conversation = []
                conversation.append(
                    {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}]
                    }
                )
                text = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = processor(text=text, return_tensors='pt')
                input_ids = inputs["input_ids"]
                pixel_values = None
                attention_mask = inputs["attention_mask"]
            else:
                conversation = []
                conversation.append(
                    {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}] + \
                                [{"type": "image"}] * 1
                    }
                )
                text = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = processor(images=[image], text=text, return_tensors='pt')
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                attention_mask = inputs["attention_mask"]
                
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values.to(dtype=model.dtype, device='cuda', non_blocking=True) if pixel_values is not None else None,
                    attention_mask=attention_mask.to(device='cuda'),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "llava-interleave" in args.model_base:
                assistant_index = outputs.find("assistant\n")
                outputs = outputs[assistant_index+len("assistant\n"):].strip()
            elif "llava-1.5" in args.model_base:
                assistant_index = outputs.find("ASSISTANT:")
                outputs = outputs[assistant_index+len("ASSISTANT:"):].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": photo_id,
                                    "prompt": cur_prompt,
                                    "answer": row["Correct Answer"], 
                                    "response": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vision_input", type=bool, default=False)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--directory", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)