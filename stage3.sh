CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --multi_gpu Stage3.py \
  --llm_name meta-llama/Meta-Llama-3-8B \
  --max_tokens 8000
