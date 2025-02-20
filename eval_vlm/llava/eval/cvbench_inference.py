import argparse
import torch
import os
import json
import math
from tqdm import tqdm
import shortuuid
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

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

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, processor, single_pred_prompt=False, no_vision_input=False):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.processor = processor
        self.single_pred_prompt = single_pred_prompt
        self.no_vision_input = no_vision_input

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["prompt"]
        if self.single_pred_prompt:  # MCQA format
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        image_file = line["image"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        conversation = []
        if self.no_vision_input:
            conversation.append(
                {
                "role": "user",
                "content": [{"type": "text", "text": qs}]
                }
            )
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=text, return_tensors='pt')
            return inputs["input_ids"], None, inputs["attention_mask"]
        else:
            conversation.append(
                {
                "role": "user",
                "content": [{"type": "text", "text": qs}] + \
                            [{"type": "image"}] * 1
                }
            )
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            inputs = self.processor(images=[image], text=text, return_tensors='pt')
            return inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]
        
    def __len__(self):
        return len(self.questions)

def collate_fn_llava(batch):
    input_ids, pixel_values, attention_mask = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    if pixel_values[0]==None:
        pixel_values = None
    else:
        pixel_values = torch.stack(pixel_values, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return input_ids, pixel_values, attention_mask

def create_data_loader(model_base, questions, image_folder, tokenizer, image_processor, model_config, processor, single_pred_prompt=False, no_vision_input=False, batch_size=1, num_workers=16):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, processor, single_pred_prompt, no_vision_input)
    if "llava-1.5" in model_base or "llava-interleave" in model_base:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_llava)
    return data_loader

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model, processor = load_llava(model_path, args.model_base, dtype="fp16" if args.fp16 else "fp32")
    model.cuda()
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    with open(answers_file, "w") as ans_file:
        data_loader = create_data_loader(args.model_base, questions, args.image_folder, tokenizer, image_processor, model.config, processor, args.single_pred_prompt, args.no_vision_input)
        if "llava-1.5" in args.model_base or "llava-interleave" in args.model_base:
            for (input_ids, pixel_values, attention_mask), line in tqdm(zip(data_loader, questions), total=len(questions)):
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)
                if pixel_values is not None:
                    pixel_values = pixel_values.squeeze(0)
                idx = line["idx"]
                cur_prompt = line["prompt"]
                if args.single_pred_prompt:
                    cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
                source = line["source"]
                task = line["task"]
                answer = line["answer"]
                
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
                    outputs = outputs[assistant_index+len("assistant\n"):]
                elif "llava-1.5" in args.model_base:
                    assistant_index = outputs.find("ASSISTANT:")
                    outputs = outputs[assistant_index+len("ASSISTANT:"):].strip()
                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({"question_id": idx,
                                        "source": source,
                                        "task": task,
                                        "prompt": cur_prompt,
                                        "response": outputs,
                                        "answer": answer,
                                        "answer_id": ans_id,
                                        "model_id": model_name,
                                        "metadata": {}}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vision_input", type=bool, default=False)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--single-pred-prompt", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
