import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
import torch
from data_utils import load_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate import infer_auto_device_map, dispatch_model
import csv

accelerator = Accelerator()
device = accelerator.device



parser = argparse.ArgumentParser(description="Generate response and score it with a reward model")
parser.add_argument("--llm_name", type=str, default='meta-llama/Meta-Llama-3-8B', help="Base LLM model name")
parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum tokens to generate")
args = parser.parse_args()
llm_name = args.llm_name
max_tokens = args.max_tokens
num_devices = accelerator.num_processes
rank = accelerator.process_index
device_map={'':rank}
tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(llm_name, device_map = device_map, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.config.pad_token_id = model.config.eos_token_id

print('rank = {}'.format(rank))
load_file = f"Stage2_GPU={rank}.jsonl"
save_file = f"Stage3_GPU={rank}.jsonl"

all_results = []
processed_indices = []
with open(load_file, "r") as f:
    for line in f:
        all_results.append(json.loads(line))
with open(save_file, "r") as f:
    for line in f:
        processed_indices.append(json.loads(line)['index'])
    

for item in all_results:
    index = item["index"]
    if index in processed_indices:
        print(f"Skipping index {index} as it is already processed.")
        continue
    prompt = item["prompt"]
    responses = item["responses"]
    scores = item["scores"]
    result = []
    num_generated_tokens=0
    dataloader = DataLoader((responses), batch_size=20, shuffle=False)  # adjust batch_size as needed
    for batch in tqdm(dataloader,ncols=80, dynamic_ncols=True):
        try:
            with torch.no_grad():
                inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
                inputs_lengths = (inputs.input_ids != tokenizer.pad_token_id).sum()
                output_sequences = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_k = 50, top_p=1.0, temperature=0.7, num_return_sequences=1)
                result+=(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
                generated_lengths = (output_sequences != tokenizer.pad_token_id).sum()
                num_generated_tokens += ((generated_lengths - inputs_lengths))
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[GPU {rank}] OOM on batch of size {len(batch)} — retrying with batch size = 1")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            for p in batch:
                try:
                    with torch.no_grad():
                        single_input = tokenizer([p], return_tensors="pt", padding=True).to(model.device)
                        input_length = (single_input.input_ids != tokenizer.pad_token_id).sum()

                        output_seq = model.generate(
                        **single_input,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=1.0,
                        temperature=0.7,
                        num_return_sequences=1
                        )

                    decoded = tokenizer.decode(output_seq[0], skip_special_tokens=True)
                    result.append(decoded)
                    gen_length = (output_seq[0] != tokenizer.pad_token_id).sum()
                    num_generated_tokens += (gen_length - input_length).item()

                except torch.cuda.OutOfMemoryError:
                    print(f"[GPU {rank}] Still OOM on single input. Skipping.")
                    torch.cuda.empty_cache()
                    continue
    with open(save_file, "a") as f:
        record = {"index": index, "prompt": prompt, "responses": result}
        f.write(json.dumps(record) + "\n")
