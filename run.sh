#!/bin/bash
export NUMEXPR_MAX_THREADS=96
CUDA_VISIBLE_DEVICES=4 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-0 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 0 &
CUDA_VISIBLE_DEVICES=5 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-1 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 1 &
CUDA_VISIBLE_DEVICES=6 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-2 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 2 &
CUDA_VISIBLE_DEVICES=7 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-3 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 3 &

wait

CUDA_VISIBLE_DEVICES=4 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-4 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 4 &
CUDA_VISIBLE_DEVICES=5 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-5 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 5 &
CUDA_VISIBLE_DEVICES=6 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-6 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 6 &
CUDA_VISIBLE_DEVICES=7 python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-405B --save_path Hessians-Llama-31-405B-6144-8k-seed-7 --batch_size 2 --devset_size 768 --ctx_size 8192 --act_save_rate 500 --sample_proc 1 --seed 7 &
