python train_vq.py \
--dataset_dir datasets/processed/PMD_samples \
--text_dataset_dir datasets/ \
--out_dir ./vq_output \
--epochs 100 \
--warmup_epochs 10 \
--learning_rate 2e-5 \
--grad_clip 0.5 \
--dead_code_threshold 5.0