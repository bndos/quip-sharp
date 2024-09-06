#!/bin/bash
export NUMEXPR_MAX_THREADS=96
for seed in {0..3}
do
  CUDA_VISIBLE_DEVICES=7 python -m quantize_llama.hessian_offline_llama \
    --base_model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --save_path Hessians-Llama-31-405B-Instruct-6144-8k-seed-${seed} \
    --batch_size 1 --devset_size 1536 --ctx_size 8192 \
    --act_save_rate 500 --sample_proc 192 --seed ${seed}
done
