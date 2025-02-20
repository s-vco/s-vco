import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import os
import json
from PIL import Image
from chat_template_monkey_patch import apply_chat_template
from transformers.utils import logging
logger = logging.get_logger(__name__)

## JSON - LOAD/DUMP: forked from https://github.com/tatsu-lab/stanford_alpaca
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f
def jdump(obj, f, mode="w", indent=4, default=str):
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()
def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


class DatasetSVCOForLlavaInterleave(Dataset):
    def __init__(
        self,
        data_path: str,
        processor,
        tokenizer,
        max_seq_len: int,
        ignore_token_id: int = -100,
        mask_question_tokens: bool = True
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.list_data_dict = jload(data_path)
        self.max_seq_len = max_seq_len
        self.IGNORE_TOKEN_ID = ignore_token_id
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.mask_question_tokens = mask_question_tokens

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index: int):
        instance_dict_info = {}
        max_len = self.max_seq_len
        instance = self.list_data_dict[index]

        # 1. has image: chosen + rejected
        for k in ["chosen", "rejected"]:
            images = [
                Image.open(img_path).convert("RGB")
                for img_path in instance[f"{k}_images"]
            ]
            vision_inputs = {}
            if images:
                vision_inputs = self.processor.image_processor(images, return_tensors="pt")

            system_prompt = instance["system_prompt"]
            instruction_prompt = instance["instruction_prompt"]
            response = instance[f"{k}_response"]

            conversation = []
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": instruction_prompt}]
                          + [{"type": "image"}] * len(images)
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

            temp = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False
            )
            cur_input_ids = temp["input_ids"]
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]

            cur_labels = cur_input_ids.clone()
            if self.mask_question_tokens:
                masks = torch.tensor(temp["assistant_masks"][:max_len], dtype=torch.bool).unsqueeze(0)
                cur_labels = torch.where(masks, cur_labels, self.IGNORE_TOKEN_ID)

            seq_len = cur_input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_pad = torch.full(
                    (cur_input_ids.shape[0], pad_len),
                    self.PAD_TOKEN_ID,
                    dtype=cur_input_ids.dtype,
                    device=cur_input_ids.device
                )
                label_pad = torch.full(
                    (cur_labels.shape[0], pad_len),
                    self.IGNORE_TOKEN_ID,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device
                )
                cur_input_ids = torch.cat([cur_input_ids, input_pad], dim=1)
                cur_labels = torch.cat([cur_labels, label_pad], dim=1)

            instance_dict_info[f"{k}_pixel_values"] = vision_inputs["pixel_values"]
            instance_dict_info[f"{k}_input_ids"] = cur_input_ids
            instance_dict_info[f"{k}_labels"] = cur_labels
            instance_dict_info[f"{k}_attention_masks"] = cur_input_ids.ne(self.PAD_TOKEN_ID)

        # 2. no image: chosen + rejected
        system_prompt = instance["system_prompt"]
        instruction_prompt = instance["instruction_prompt"]
        for k in ["chosen", "rejected"]:
            response = instance[f"{k}_response"]

            conversation = []
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": instruction_prompt}]
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

            temp = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False
            )
            cur_input_ids = temp["input_ids"]
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]

            cur_labels = cur_input_ids.clone()
            if self.mask_question_tokens:
                masks = torch.tensor(temp["assistant_masks"][:max_len], dtype=torch.bool).unsqueeze(0)
                cur_labels = torch.where(masks, cur_labels, self.IGNORE_TOKEN_ID)

            seq_len = cur_input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_pad = torch.full(
                    (cur_input_ids.shape[0], pad_len),
                    self.PAD_TOKEN_ID,
                    dtype=cur_input_ids.dtype,
                    device=cur_input_ids.device
                )
                label_pad = torch.full(
                    (cur_labels.shape[0], pad_len),
                    self.IGNORE_TOKEN_ID,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device
                )
                cur_input_ids = torch.cat([cur_input_ids, input_pad], dim=1)
                cur_labels = torch.cat([cur_labels, label_pad], dim=1)

            instance_dict_info[f"no_img_{k}_input_ids"] = cur_input_ids
            instance_dict_info[f"no_img_{k}_labels"] = cur_labels
            instance_dict_info[f"no_img_{k}_attention_masks"] = cur_input_ids.ne(self.PAD_TOKEN_ID)

        out = tuple(
            instance_dict_info[f"{k}_{item}"]
            for k in ["chosen", "rejected"]
            for item in ["pixel_values", "input_ids", "labels", "attention_masks"]
        )
        out += tuple(
            instance_dict_info[f"no_img_{k}_{item}"]
            for k in ["chosen", "rejected"]
            for item in ["input_ids", "labels", "attention_masks"]
        )
        return out


class DatasetSVCOForLlava15(Dataset):
    def __init__(
        self,
        data_path: str,
        processor,
        tokenizer,
        max_seq_len: int,
        ignore_token_id: int = -100,
        mask_question_tokens: bool = True
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.list_data_dict = jload(data_path)
        self.max_seq_len = max_seq_len
        self.IGNORE_TOKEN_ID = ignore_token_id
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.mask_question_tokens = mask_question_tokens

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index: int):
        instance_dict_info = {}
        max_len = self.max_seq_len

        # include the last token in labels
        self.tokenizer.apply_chat_template = apply_chat_template.__get__(self.tokenizer)

        instance = self.list_data_dict[index]

        # 1. has image: chosen + rejected
        for k in ["chosen", "rejected"]:
            images = [
                Image.open(img_path).convert("RGB")
                for img_path in instance[f"{k}_images"]
            ]
            vision_inputs = {}
            if images:
                vision_inputs = self.processor.image_processor(images, return_tensors="pt")

            system_prompt = instance["system_prompt"]
            instruction_prompt = instance["instruction_prompt"]
            response = instance[f"{k}_response"]

            conversation = []
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": instruction_prompt}]
                          + [{"type": "image"}] * len(images)
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

            temp = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False
            )
            cur_input_ids = temp["input_ids"]
            assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)
            
            # include the last token in labels
            assistant_masks[0, -1] = True

            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                assistant_masks = assistant_masks[:, :max_len]

            cur_labels = cur_input_ids.clone()
            if self.mask_question_tokens:
                cur_labels = torch.where(assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)

            seq_len = cur_input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_pad = torch.full(
                    (cur_input_ids.shape[0], pad_len),
                    self.PAD_TOKEN_ID,
                    dtype=cur_input_ids.dtype,
                    device=cur_input_ids.device
                )
                label_pad = torch.full(
                    (cur_labels.shape[0], pad_len),
                    self.IGNORE_TOKEN_ID,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device
                )
                cur_input_ids = torch.cat([cur_input_ids, input_pad], dim=1)
                cur_labels = torch.cat([cur_labels, label_pad], dim=1)

            instance_dict_info[f"{k}_pixel_values"] = vision_inputs["pixel_values"] if images else torch.zeros((cur_input_ids.shape[0], 3, 224, 224))
            instance_dict_info[f"{k}_input_ids"] = cur_input_ids
            instance_dict_info[f"{k}_labels"] = cur_labels
            instance_dict_info[f"{k}_attention_masks"] = cur_input_ids.ne(self.PAD_TOKEN_ID)

        # 2. no image: chosen + rejected
        system_prompt = instance["system_prompt"]
        instruction_prompt = instance["instruction_prompt"]
        for k in ["chosen", "rejected"]:
            response = instance[f"{k}_response"]

            conversation = []
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": instruction_prompt}]
            })
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

            temp = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False
            )
            cur_input_ids = temp["input_ids"]
            assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)
            
            # include the last token in labels
            assistant_masks[0, -1] = True

            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                assistant_masks = assistant_masks[:, :max_len]

            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                cur_labels = torch.where(assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)

            seq_len = cur_input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_pad = torch.full(
                    (cur_input_ids.shape[0], pad_len),
                    self.PAD_TOKEN_ID,
                    dtype=cur_input_ids.dtype,
                    device=cur_input_ids.device
                )
                label_pad = torch.full(
                    (cur_labels.shape[0], pad_len),
                    self.IGNORE_TOKEN_ID,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device
                )
                cur_input_ids = torch.cat([cur_input_ids, input_pad], dim=1)
                cur_labels = torch.cat([cur_labels, label_pad], dim=1)

            instance_dict_info[f"no_img_{k}_input_ids"] = cur_input_ids
            instance_dict_info[f"no_img_{k}_labels"] = cur_labels
            instance_dict_info[f"no_img_{k}_attention_masks"] = cur_input_ids.ne(self.PAD_TOKEN_ID)

        out = tuple(
            instance_dict_info[f"{k}_{item}"]
            for k in ["chosen", "rejected"]
            for item in ["pixel_values", "input_ids", "labels", "attention_masks"]
        )
        out += tuple(
            instance_dict_info[f"no_img_{k}_{item}"]
            for k in ["chosen", "rejected"]
            for item in ["input_ids", "labels", "attention_masks"]
        )
        return out


def collate_fn_svco_llava(batch):
    (
        chosen_pixel_values,
        chosen_input_ids,
        chosen_labels,
        chosen_attention_mask,
        rejected_pixel_values,
        rejected_input_ids,
        rejected_labels,
        rejected_attention_mask,
        no_img_chosen_input_ids,
        no_img_chosen_labels,
        no_img_chosen_attention_mask,
        no_img_rejected_input_ids,
        no_img_rejected_labels,
        no_img_rejected_attention_mask,
    ) = zip(*batch)

    return (
        torch.cat(chosen_pixel_values, dim=0),
        torch.cat(chosen_input_ids, dim=0),
        torch.cat(chosen_labels, dim=0),
        torch.cat(chosen_attention_mask, dim=0),
        torch.cat(rejected_pixel_values, dim=0),
        torch.cat(rejected_input_ids, dim=0),
        torch.cat(rejected_labels, dim=0),
        torch.cat(rejected_attention_mask, dim=0),
        torch.cat(no_img_chosen_input_ids, dim=0),
        torch.cat(no_img_chosen_labels, dim=0),
        torch.cat(no_img_chosen_attention_mask, dim=0),
        torch.cat(no_img_rejected_input_ids, dim=0),
        torch.cat(no_img_rejected_labels, dim=0),
        torch.cat(no_img_rejected_attention_mask, dim=0),
    )
    

def create_svco_data_loader(model_type, data_path, processor, tokenizer, max_seq_len, image_token_index, ignore_token_id, mask_question_tokens):
    if model_type == "llava-interleave":
        dataset = DatasetSVCOForLlavaInterleave(data_path, processor, tokenizer, max_seq_len, ignore_token_id, mask_question_tokens)
        collator = collate_fn_svco_llava
    elif model_type == "llava-1.5":
        dataset = DatasetSVCOForLlava15(data_path, processor, tokenizer, max_seq_len, ignore_token_id, mask_question_tokens)
        collator = collate_fn_svco_llava
    return dataset, collator
