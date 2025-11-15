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


# 참고
# of samples: 80000
block_size of PickleLoader in dataset.py: 중간 토큰의 개수, 길이가 짧을 경우 패딩으로 채움
sequence_unit in dataset.py: 1개 토큰을 이루는 샘플의 개수. 
time_embed in model_neural_transformer.py: 트랜스포머에게 전달할 시간정보. 토큰의 길이가 400이고 time_embed가 200이라면, 1~200까지와 201~400까지의 시간정보가 동일

ex) 80000만 샘플, 200 샘플마다 1개 토큰으로 바꾼다면 --> 입력은 [1, 400, 200]이고, 여기서 400이 block_size임. block_size의 최댓값이 1024라면, 624개의 패딩 존재.

여기서 time_embed가 100이라면, 0~99, 100~199, 200~299, ... 들이 서로 같은 시간정보를 가짐. 따라서, time_embed를 400 이상으로 해야 함.

Transformer 기반 VQ: sequence_unit = 200을 중간 latent space = 128로 바꾼 뒤, 다시 복원하는 transformer임.