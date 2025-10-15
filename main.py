import argparse
import torch
from utils import ArmoRMPipeline
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

accelerator = Accelerator()
device = accelerator.device



parser = argparse.ArgumentParser(description="Generate response and score it with a reward model")
parser.add_argument("--llm_name", type=str, default='meta-llama/Meta-Llama-3-8B', help="Base LLM model name")
parser.add_argument("--rm_name", type=str, default='RLHFlow/ArmoRM-Llama3-8B-v0.1', help="Reward model name")
parser.add_argument("--dataset", type=str, default="./datasets/alpaca_farm_eval.json", help="Path to the dataset")
parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum tokens to generate")
parser.add_argument("--tau", type=int, default=256, help="numer of tokens for speculative rejection")
parser.add_argument("--n", type=int, default=120, help="numer of generated response")
parser.add_argument("--a", type=float, default=0.5, help="percentage of rejections")
args = parser.parse_args()
llm_name = args.llm_name
rm_name = args.rm_name
dataset = args.dataset
max_tokens = args.max_tokens
tau = args.tau
n = args.n
a = args.a

#torch.cuda.empty_cache()
#pipeline = transformers.pipeline(
#  "text-generation",
#  model=llm_name,
#  model_kwargs={"torch_dtype": torch.float32},
#  device_map='auto'
#)
num_devices = accelerator.num_processes
rank = accelerator.process_index
device_map={'':rank}
tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(llm_name, device_map = device_map, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map = device_map, trust_remote_code=True, torch_dtype=torch.bfloat16)

# maps model layers automatically
#print(device_map)
#model = dispatch_model(model, device_map=device_map)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.config.pad_token_id = model.config.eos_token_id
print('rank = {}'.format(rank))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_GPU="+str(rank))
filename = f"records_{timestamp}.json"
prompts = load_prompts(dataset)
prompts = prompts[rank::num_devices]
model.eval()
with open(filename, "a") as f:
    
    for index, prompt in enumerate(prompts):
        print('=======processing prompt: ', index,'=========')
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        num_tokens = inputs.input_ids.shape[1]
        num_generated_tokens = 0
        #Response Generation
        output_sequences = model.generate(**inputs, max_new_tokens=tau, do_sample=True, top_k = 50, top_p=1.0, temperature=0.7, num_return_sequences=n)
        generated_lengths = (output_sequences != tokenizer.pad_token_id).sum(dim=1)
        num_generated_tokens = sum((generated_lengths - num_tokens).tolist())
        result = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        
        #score Generation
        score = []
        for response in tqdm(result):
            l = len(prompt)
            score.append(rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response[l+1:]}]))
            
        #sort score and keeping alpha*100% of the lowest score
        ziplist = list(zip(result, score))
        ziplist = sorted(ziplist, key = lambda x: x[1], reverse=True)
        p = [ziplist[i][0] for i in range(int(n*a))]
        loader = DataLoader(p, batch_size=4, shuffle=False)
        del inputs
        del output_sequences
        del result
        del score
        result = []
        torch.cuda.empty_cache()
        #Generate the final response and score them
        for batch in tqdm(loader):
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            inputs_lengths = (inputs.input_ids != tokenizer.pad_token_id).sum()
            output_sequences = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_k = 50, top_p=1.0, temperature=0.7, num_return_sequences=1)
            result+=(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
            generated_lengths = (output_sequences != tokenizer.pad_token_id).sum()
            num_generated_tokens += ((generated_lengths - inputs_lengths))
            del inputs
            del output_sequences
            torch.cuda.empty_cache()
            torch.cuda.memory_summary()
            
        score = []
        for response in tqdm(result):
            l = len(prompt)
            score.append(rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response[l+1:]}]))
        best = score.index(max(score))
        print(num_generated_tokens)
        record = {
            "prompt": prompt,
            "best response": result[best],
            "score": score[best],
            "num_tokens": num_generated_tokens.item(),
            "avg_num_tokens": float(num_generated_tokens.item())/n
        }
        f.write(json.dumps(record) + "\n")
        torch.cuda.empty_cache()

