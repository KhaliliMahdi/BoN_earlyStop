import argparse
import torch
import torch
from data_utils import load_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import ArmoRMPipeline
import torch
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device



parser = argparse.ArgumentParser(description="Generate response and score it with a reward model")
parser.add_argument("--rm_name", type=str, default='RLHFlow/ArmoRM-Llama3-8B-v0.1', help="Reward model name")
parser.add_argument("--a", type=float, default=0.5, help="percentage of rejections")

args = parser.parse_args()
rm_name = args.rm_name
a = args.a


num_devices = accelerator.num_processes
rank = accelerator.process_index
device_map={'':rank}
rm = ArmoRMPipeline(rm_name, device_map = device_map, trust_remote_code=True, torch_dtype=torch.bfloat16)

print('rank = {}'.format(rank))
load_file = f"Stage1_GPU={rank}.jsonl"
save_file = f"Stage2_GPU={rank}.jsonl"

all_results = []
with open(load_file, "r") as f:
    for line in f:
        all_results.append(json.loads(line))

score = []
for item in all_results:
    index = item["index"]
    prompt = item["prompt"]
    responses = item["responses"]
    n = len(responses)
    for response in tqdm(responses,ncols=80, dynamic_ncols=True):
        l = len(prompt)
        score.append(rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response[l+1:]}]))
    ziplist = list(zip(responses, score))
    ziplist = sorted(ziplist, key = lambda x: x[1], reverse=True)
    r = [ziplist[i][0] for i in range(int(n*a))]
    s = [ziplist[i][1] for i in range(int(n*a))]
    with open(save_file, "a") as f:
        record = {"index": index, "prompt": prompt, "responses": r, "scores": s}
        f.write(json.dumps(record) + "\n")