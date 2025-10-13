import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import ArmoRMPipeline



parser = argparse.ArgumentParser(description="Generate response and score it with a reward model")
parser.add_argument("--llm_name", type=str, required=True, help="Base LLM model name (e.g., Meta-Llama-3-8B)")
parser.add_argument("--rm_name", type=str, required=True, help="Reward model name (e.g., ArmoRM-Llama3-8B-v0.1)")
parser.add_argument("--prompt", type=str, default="Explain why ETFs are popular for beginner investors.", help="Prompt to generate response for")
parser.add_argument("--max_tokens", type=int, default=150, help="Maximum tokens to generate")
args = parser.parse_args()


import transformers
import torch
pipeline = transformers.pipeline(
  "text-generation",
  model=args.llm_name,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device = 0 if torch.cuda.is_available() else -1,
)
prompt = "This restaurant is awesome"
result = pipeline(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
response1 = result[0]['generated_text']
print(response1)
rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": 'hellow howaryou'}])
print(score1)
