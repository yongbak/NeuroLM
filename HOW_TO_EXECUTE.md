It needs processed .pkl and .bin.

1. .pkl is actual datasets
run python dataset_maker/prepare_from_txt_signal.py
or
run python dataset_maker/create_dummy_data.py

.pkl datasets are saved at datasets/processed/PMD-samples


2. .bin is binary expression of an natural language dataset
# https://huggingface.co/datasets/Skylion007/openwebtext
run python text_dataset_maker/prepare.py
or
just load pre-processed REPO_ROOT/datasets/text/train.bin and REPO_ROOT/datasets/text/val.bin


3. run train.sh
python train_vq.py --dataset_dir datasets/processed/PMD_samples --text_dataset_dir datasets/ --out_dir ./vq_output