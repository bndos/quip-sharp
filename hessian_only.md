```bash
python -m quantize_llama.hessian_offline_llama --base_model meta-llama/Meta-Llama-3.1-8B --save_path Hessians-Llama-31-8B-6144-8k --batch_size 4 --devset_size 6144 --ctx_size 8192 --act_save_rate 50 --sample_proc 96 --seed 1
```

