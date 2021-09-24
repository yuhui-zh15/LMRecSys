CUDA_VISIBLE_DEVICES=2 python infer.py --model_name_or_path t5-small --eval_batch_size 1
CUDA_VISIBLE_DEVICES=2 python infer.py --model_name_or_path t5-base --eval_batch_size 1
CUDA_VISIBLE_DEVICES=2 python infer.py --model_name_or_path t5-3b --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python infer.py --model_name_or_path t5-large --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python infer.py --model_name_or_path t5-medium --eval_batch_size 1
