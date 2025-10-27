CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --multi_gpu Stage1.py \
  --llm_name meta-llama/Meta-Llama-3-8B \
  --dataset ./datasets/alpaca_farm_eval.json \
  --n 120 \
  --tau 256
