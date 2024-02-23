import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--peft_path', type=str, required=True)
parser.add_argument('--model_name', type=str, default='/gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf')
args = parser.parse_args()

base_model = LlamaForCausalLM.from_pretrained(
    args.model_name,
    return_dict=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
peft_model = PeftModel.from_pretrained(base_model, args.peft_path)
model = peft_model.merge_and_unload()
tokenizer = LlamaTokenizer.from_pretrained(args.model_name, legacy=False)

output_dir = args.peft_path + '/merged'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")