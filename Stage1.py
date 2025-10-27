import argparse
import torch
import torch
from data_utils import load_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device



parser = argparse.ArgumentParser(description="Generate response and score it with a reward model")
parser.add_argument("--llm_name", type=str, default='meta-llama/Meta-Llama-3-8B', help="Base LLM model name")
parser.add_argument("--dataset", type=str, default="./datasets/alpaca_farm_eval.json", help="Path to the dataset")
parser.add_argument("--tau", type=int, default=256, help="numer of tokens for speculative rejection")
parser.add_argument("--n", type=int, default=120, help="numer of generated response")
args = parser.parse_args()
llm_name = args.llm_name
dataset = args.dataset
tau = args.tau
n = args.n


num_devices = accelerator.num_processes
rank = accelerator.process_index
device_map={'':rank}
tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(llm_name, device_map = device_map, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)



tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.config.pad_token_id = model.config.eos_token_id
print('rank = {}'.format(rank))
filename = f"Stage1_GPU={rank}.json"
prompts, index = load_prompts(dataset)
indices = index[rank::num_devices]
prompts = prompts[rank::num_devices]
model.eval()



for index, prompt in zip(indices,prompts):
    print('=======processing prompt: ', index,'=========')
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    num_tokens = inputs.input_ids.shape[1]
    num_generated_tokens = 0
        #Response Generation
    output_sequences = model.generate(**inputs, max_new_tokens=tau, do_sample=True, top_k = 50, top_p=1.0, temperature=0.7, num_return_sequences=n)
    generated_lengths = (output_sequences != tokenizer.pad_token_id).sum(dim=1)
    num_generated_tokens = sum((generated_lengths - num_tokens).tolist())
    result = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    record = {"index": index, 'num_tokens': num_generated_tokens, "prompt": prompt, "responses": result}
    with open(filename, "a") as f:
        f.write(json.dumps(record) + "\n")