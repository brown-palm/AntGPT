# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training, PeftModel

from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    train_distill,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

def build_student_model(device):
    
    llama_config = LlamaConfig(vocab_size = 32000,
                        hidden_size = 768,
                        intermediate_size = 2048,
                        num_hidden_layers = 6,
                        num_attention_heads = 6,
                        num_key_value_heads = 6,
                        max_position_embeddings = 300,
                        )

    model = LlamaForCausalLM(llama_config)
    model.to(device)
    return model

def main(peft_model: str=None,
         **kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained teacher_model and setup its configuration
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained teacher_model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        teacher_model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            teacher_model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            with torch.device("meta"):
                teacher_model = LlamaForCausalLM(llama_config)

    else:
        teacher_model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            teacher_model = BetterTransformer.transform(teacher_model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(teacher_model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the teacher_model for int8 training if quantization is enabled
    if train_config.quantization:
        teacher_model = prepare_model_for_int8_training(teacher_model)

    # Convert the teacher_model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        teacher_model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    if peft_model:
        print("loading peft model")
        teacher_model = PeftModel.from_pretrained(teacher_model, peft_model)

    #setting up FSDP if enable_fsdp is enabled

    if not train_config.quantization and not train_config.enable_fsdp:
        teacher_model.to("cuda")

    device = teacher_model.device
    student_model = build_student_model(device)
    
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
        use_goal=train_config.use_goal,
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
        use_goal=train_config.use_goal,
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    print(kwargs)
    shuffle = 'shuffle' in kwargs
    print("shuffle training set: ", shuffle)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
        shuffle=shuffle
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
    else:
        eval_dataloader = None

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            student_model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            student_model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    # freeze the teacher_model
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Start the training process
    results = train_distill(
        student_model,
        teacher_model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)

    """
    CUDA_VISIBLE_DEVICES=1 python distill.py --quantization --model_name peft_ckpt/ego4d_old/ego4d_v1/lora/7B/1/merged --dataset ego4d_v1 --num_epochs=20 --run_validation=True --batch_size_training=32 val_batch_size=128 lr=10e-4 -output_dir ft_ckpt/ego4d_old/ego4d_v1/dist_1:0.5_layer6_bs32_10e-4 --mlm_coef 0.5
    """