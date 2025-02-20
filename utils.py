import torch.distributed as dist
import pprint

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)

def rank0_pprint(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(*args)

def rank0_print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(_)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

