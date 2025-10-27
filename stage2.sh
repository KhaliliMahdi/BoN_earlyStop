CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --multi_gpu Stage2.py \
  --rm_name RLHFlow/ArmoRM-Llama3-8B-v0.1 \
  --a 0.5

