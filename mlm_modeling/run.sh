python -m torch.distributed.launch \
    --nproc_per_node 4 run_mlm.py \
    --model_name_or_path bert-base-cased \
    --train_file movie_data.txt \
    --validation_file movie_data.txt \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --output_dir runs/bert_base_cased_512bs \
    --preprocessing_num_workers 32 \
    --pad_to_max_length \
    --gradient_checkpointing \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512
    