import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from transformers import AutoProcessor, Idefics2ForConditionalGeneration, AutoModelForVision2Seq
from transformers.integrations import deepspeed

from dataset import SupervisedDataset, data_collator
from transformers import Trainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="HuggingFaceM4/Idefics3-8B-Llama3")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = ['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)


def make_supervised_data_module(
    processor,
    data_args,
    data_collator=None,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        processor,
        max_length=max_length,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            processor
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, processor=processor, max_length=max_length),
    )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0



def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, 
        do_image_splitting=False
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        device_map=device_map,
        _attn_implementation="flash_attention_2"
    )

    model.config.use_cache = False

    if not training_args.tune_vision:
        model.model.vision_model.requires_grad_(False)
    if not training_args.tune_llm:
        model.model.text_model.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the Idefics2 model.")
        for name, param in model.model.text_model.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','connector']
        if training_args.tune_vision:
            modules_to_save.append('vision_model')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.model.text_model.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))



    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        data_collator=data_collator,
        max_length=training_args.model_max_length,
    )
    
    #training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module,  # data_collator, train_dataset, and (optionally) eval_dataset
    )


    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    train()
