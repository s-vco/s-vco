import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
import transformers
from accelerate.utils import DistributedType
from transformers import AutoProcessor, LlavaForConditionalGeneration

from data_collator import create_svco_data_loader
from trainer import SVCOTrainerForLlava
from utils import rank0_print, rank0_pprint


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "llava-1.5": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "llava-interleave": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    }
}

@dataclass
class ModelArguments:
    model_family_id: str = field(default=None)
    model_name_or_path: str = field(default=None)
    dataset_path: str = field(default=None)
    eval_dataset_path: Optional[str] = field(default=None)
    vision_feature_select_strategy: str = field(default=None)
    patch_size: int = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    svco: Optional[bool] = field(default=False)
    use_flash_attn: Optional[bool] = field(default=True)
    deepspeed: Optional[str] = field(default="zero3")
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: Optional[str] = field(default=False)
    model_max_length: int = field(default=2048)
    fix_vit: bool = True
    fix_multimodal_projector: bool = False
    beta_img_win_vs_no_img_preference: Optional[float] = field(default=0.1)
    beta_no_img_vs_img_lose_preference: Optional[float] = field(default=0.1)
    beta_img_lose_vs_no_img_preference: Optional[float] = field(default=0.1)
    beta_no_img_vs_img_win_preference: Optional[float] = field(default=0.1)
    generate_during_eval: bool = field(default=False)
    output_dir: str = field(default="./debug_output/") 
    deepspeed: str = field(default="./ds_configs/zero3.json") 
    reference_free: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=42)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit
            lora_module_names.remove(name)
    return list(lora_module_names)


def train():
    os.environ["WANDB_PROJECT"] = "S-VCO"

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # DS & DDP
    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    device_map = None
    
    loading_kwargs = dict(
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        # low_cpu_mem_usage=True
    )
    if training_args.use_flash_attn:
        loading_kwargs["attn_implementation"] = "flash_attention_2"
    # loading_kwargs["embd_pdrop"] = 0  # don't set the dropout rate for embeddings to 0 (no dropout).
    
    if model_args.model_family_id == "llava-1.5":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **loading_kwargs
            )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, 
                                        vision_feature_select_strategy=model_args.vision_feature_select_strategy, 
                                        patch_size=model_args.patch_size,
                                        add_eos_token=True)  # necessary
    elif model_args.model_family_id == "llava-interleave":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            **loading_kwargs
            )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, 
                                        vision_feature_select_strategy="default",  # lv-interleave newly updated hf-repo
                                        patch_size=model_args.patch_size)    
    model.config.hidden_size = model.language_model.config.hidden_size  # useful for deepspeed
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    
    # grad-checkpointing --> more time for less mem.
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # freeze certain params
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if training_args.fix_vit: 
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)
    
    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if training_args.fix_multimodal_projector: 
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # print trainable parameters
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # load data
    rank0_print("Loading data...")
    if training_args.svco:
        train_dataset, data_collator = create_svco_data_loader(model_type=model_args.model_family_id,
                                                    data_path=model_args.dataset_path, 
                                                    processor=processor, 
                                                    tokenizer=tokenizer, 
                                                    max_seq_len=training_args.model_max_length, 
                                                    image_token_index=model.config.image_token_index,
                                                    ignore_token_id=-100,
                                                    mask_question_tokens=True, 
                                                    )
        eval_dataset, data_collator = create_svco_data_loader(model_type=model_args.model_family_id,
                                                    data_path=model_args.eval_dataset_path, 
                                                    processor=processor, 
                                                    tokenizer=tokenizer, 
                                                    max_seq_len=training_args.model_max_length, 
                                                    image_token_index=model.config.image_token_index,
                                                    ignore_token_id=-100,
                                                    mask_question_tokens=True, 
                                                    )

    # print args
    rank0_print("==== model args ====")
    rank0_pprint(model_args)
    rank0_print("==== training args ====")
    rank0_pprint(training_args)

    # manually create reference model --> for ds zero-3
    # otherwise ERROR: trl/dpo_trainer: DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoCausalLM.from_pretrained()`.
    ref_model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **loading_kwargs
        )
    ref_model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
    # freeze all params for ref_model
    parameter_names = [n for n, _ in ref_model.named_parameters()]
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model.eval() # ref_model -> eval mode

    # start trainner
    if training_args.svco:
        if model_args.model_family_id == "llava-interleave" or model_args.model_family_id == "llava-1.5":
            trainer = SVCOTrainerForLlava(
                ref_model=ref_model,
                beta_img_win_vs_no_img_preference=training_args.beta_img_win_vs_no_img_preference,
                beta_no_img_vs_img_lose_preference=training_args.beta_no_img_vs_img_lose_preference,
                beta_img_lose_vs_no_img_preference=training_args.beta_img_lose_vs_no_img_preference,
                beta_no_img_vs_img_win_preference=training_args.beta_no_img_vs_img_win_preference,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer, 
                model=model,
                max_length=training_args.model_max_length,
                peft_config=None,
                generate_during_eval=training_args.generate_during_eval,
            )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_state()

    model.config.save_pretrained(training_args.output_dir)
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir) 
    
    rank0_print(f"^_^ Final model saved to: {training_args.output_dir} ^_^")
    rank0_print("^_^ Training Complete ^_^")


if __name__ == "__main__":
    train()

