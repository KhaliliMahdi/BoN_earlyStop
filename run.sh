python main.py \
  --llm_name meta-llama/Meta-Llama-3-8B \
  --rm_name RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --dataset ./datasets/alpaca_farm_eval.json \
  --max_tokens 5000 \
  --n 10 \
  --tau 256 \
  --a 0.5

