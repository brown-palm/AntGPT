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
    train_proj,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

class ProjLlama(torch.nn.Module):
    def __init__(self, llama, llama_config, vis_size):
        super().__init__()
        self.llama = llama
        self.proj = torch.nn.Linear(vis_size, llama_config.hidden_size, bias=False)
        for param in self.llama.parameters():
            param.requires_grad = False
    
    def forward(self, obs_feats, input_ids, labels, attention_mask):
        bs = obs_feats.shape[0]
        obs_embs = self.proj(obs_feats)
        inputs_embeds = []
        for i in range(bs):        
            obs_emb = obs_embs[i]
            input_id = input_ids[i]
            # start of dummy input is the first -1 in input_ids
            dummy_start = torch.where(input_id == -1)[0][0]
            # end of dummy input is the last -1 in input_ids
            dummy_end = torch.where(input_id == -1)[0][-1]
            input_start = input_id[:dummy_start]
            input_end = input_id[dummy_end+1:]
            emb_start = self.llama.model.model.embed_tokens(input_start)
            emb_end = self.llama.model.model.embed_tokens(input_end)
            inputs_embed = torch.cat([emb_start, obs_emb, emb_end], dim=0)
            inputs_embeds.append(inputs_embed)
        inputs_embeds = torch.stack(inputs_embeds, dim=0)
        batch = {
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        outputs = self.llama(**batch)
        return outputs

    
    def generate(self, obs_feats, input_ids, labels, attention_mask, max_new_tokens=200):
        bs = obs_feats.shape[0]
        obs_emb = self.proj(obs_feats[0])
        input_id = input_ids[0]
        
        label_start = torch.where(labels[0] != -100)[0][0]
        input_id = input_id[:label_start]
        
        # start of dummy input is the first -1 in input_ids
        dummy_start = torch.where(input_id == -1)[0][0]
        # end of dummy input is the last -1 in input_ids
        dummy_end = torch.where(input_id == -1)[0][-1]
        input_start = input_id[:dummy_start]

        emb_start = self.llama.model.model.embed_tokens(input_start)
        inputs_embed = torch.cat([emb_start, obs_emb], dim=0)
        batch = {
            "inputs_embeds": inputs_embed.unsqueeze(0),
            "attention_mask": attention_mask[0][:inputs_embed.shape[0]].unsqueeze(0),
        }
        with torch.no_grad():
            outputs = self.llama.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=0.3,
                min_length=None,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1,
                num_return_sequences = 5,
                pad_token_id = 0,
            )
        return outputs
    

def main(peft_model: str=None, proj_path: str=None,
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

    # Load the pre-trained model and setup its configuration
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
        llama_config = LlamaConfig.from_pretrained(train_config.model_name)
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {
             "pad_token": "<PAD>",
            }
        )
    if peft_model:
        print("loading peft model")
        model = PeftModel.from_pretrained(model, peft_model)
    else:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)

    #setting up FSDP if enable_fsdp is enabled

    if not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    vis_model = ProjLlama(model, llama_config, vis_size=768)
    if proj_path:
        vis_model.proj.load_state_dict(torch.load(proj_path))
    vis_model.to("cuda")
    print("trainable parameter size: ", sum(p.numel() for p in vis_model.parameters() if p.requires_grad))
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
        use_vis=train_config.use_vis,
        pure_vis=train_config.pure_vis,
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
        use_vis=train_config.use_vis,
        pure_vis=train_config.pure_vis,
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
            vis_model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            vis_model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train_proj(
        vis_model,
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
