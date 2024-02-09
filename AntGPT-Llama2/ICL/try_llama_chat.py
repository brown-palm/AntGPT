import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

model_path = "/gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        # print('-' * 40)
        # print(tokenizer.decode(input_ids[0]))
        if input_ids[0][-1] == 13:
            return True

        return False


ctx = """A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits.
User: Hello, AI.
AI: Hello! How can I assist you today?
"""
print(ctx)
while True:
    print('-' * 40)
    print(ctx.rstrip("\n"))
    prompt = input(f'User: ')
    if ctx != "":
        ctx = ctx + "User: " + prompt + "\n"
    else:
        ctx = prompt + "\n"

    ctx = (ctx[-1920:]) if len(ctx) >= 2048 else ctx

    if len(ctx.strip()) > 0:
        batch = tokenizer(ctx, return_tensors="pt")
        result = model.generate(batch["input_ids"].to(model.device),
                                do_sample=True,
                                top_k=50,
                                max_length=2048,
                                top_p=0.95,
                                temperature=1.0,
                                stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub()]),
                                # repetition_penalty=1.17
                                )
        decoded = tokenizer.decode(result[0])
        ctx = decoded + "\n"
        # print("answer: ", decoded)

