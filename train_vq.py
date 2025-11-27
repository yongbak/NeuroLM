"""
Tuning by Yonghyeon Park
https://github.com/yongbak/NeuroLM
"""

from constants import (
    NUM_WORKERS,
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_TEXT_BATCH_SIZE,
    NUM_OF_TOTAL_SAMPLES,
    NUM_OF_SAMPLES_PER_TOKEN,
    NUM_OF_TOTAL_TOKENS,
    SAMPLING_RATE,
    VAE_AUGMENT_FACTOR,
    CODEBOOK_SIZE,
    DECAY,
    BETA,
    OFFLINE,
    DEBUG_ENCODER
)

import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from augmentor import VAEAugmentor, AugmentedDataset

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
from dataset import PickleLoader
from pathlib import Path
from utils import cosine_scheduler
import math

import torch.multiprocessing as mp
# CUDA와 multiprocessing을 함께 사용하기 위해 spawn method 설정
mp.set_start_method('spawn', force=True)

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

    dtype = DEFAULT_DTYPE
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    import time
    torch.manual_seed(time.time_ns())
    #torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # text data loader
    data_dir = os.path.join(args.text_dataset_dir, 'text')
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


    print('prepare dataloader...')
    train_files = list(Path(args.dataset_dir, 'train').rglob('*.pkl'))
    val_files = list(Path(args.dataset_dir, 'val').rglob('*.pkl'))
    dataset_train = PickleLoader(train_files, block_size=NUM_OF_TOTAL_TOKENS, sampling_rate=SAMPLING_RATE, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN)
    dataset_val = PickleLoader(val_files, block_size=NUM_OF_TOTAL_TOKENS, sampling_rate=SAMPLING_RATE, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN)

    ### VAE augmentor 적용
    if VAE_AUGMENT_FACTOR > 0:
        vae_augmentor = VAEAugmentor(pretrained_path='./vae_models/vae_augmentor_benign.pt')
        augmented_dataset_train = AugmentedDataset(dataset_train, vae_augmentor, num_augmentations_per_sample=VAE_AUGMENT_FACTOR, noise_scale=0.9, include_original=False)
        augmented_dataset_val = AugmentedDataset(dataset_val, vae_augmentor, num_augmentations_per_sample=VAE_AUGMENT_FACTOR, noise_scale=0.9, include_original=False)

        dataset_train = torch.utils.data.ConcatDataset([dataset_train, augmented_dataset_train])
        dataset_val = torch.utils.data.ConcatDataset([dataset_val, augmented_dataset_val])

    print('finished!')

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,                  # Tuning hyperparameter
            pin_memory=True,
            drop_last=True,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,                  # Tuning hyperparameter
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
    # validation 데이터로더
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    # model init
    # block_size는 prepare_from_txt_signal.py의 window_size // sequence_unit과 동일
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768,
                    block_size=NUM_OF_TOTAL_TOKENS, patch_size=NUM_OF_SAMPLES_PER_TOKEN, sample_size=NUM_OF_TOTAL_SAMPLES,
                    bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768,
                    block_size=NUM_OF_TOTAL_TOKENS, patch_size=NUM_OF_SAMPLES_PER_TOKEN, sample_size=NUM_OF_TOTAL_SAMPLES,
                    bias=False, dropout=0., num_classes=0, in_chans=128)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'scratch'

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        encoder_conf = NTConfig(**encoder_args)
        decoder_conf = NTConfig(**decoder_args)
        model = VQ_Align(encoder_conf, decoder_conf,
                         # Patch size, 1개 토큰이 커버하는 샘플 개수는 encoder_conf에 존재
                         n_embed=CODEBOOK_SIZE,
                         embed_dim=128,
                         decay=DECAY,  # Lowered from 0.95 to reduce past bias and prevent collapse
                         beta=BETA,    # Increased commitment loss to prevent collapse
                         offline=OFFLINE,
                         dead_code_threshold=args.dead_code_threshold)
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['encoder_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            encoder_args[k] = checkpoint_model_args[k]
        checkpoint_model_args = checkpoint['decoder_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            decoder_args[k] = checkpoint_model_args[k]
        # create the model
        encoder_conf = NTConfig(**encoder_args)
        decoder_conf = NTConfig(**decoder_args)
        model = VQ_Align(encoder_conf, decoder_conf,
                         n_embed=CODEBOOK_SIZE,
                         embed_dim=128,
                         decay=DECAY,  # Lowered from 0.95 to reduce past bias and prevent collapse
                         beta=BETA,    # Increased commitment loss to prevent collapse
                         offline=OFFLINE,
                         dead_code_threshold=args.dead_code_threshold)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_keys
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, dir=os.path.join(args.out_dir, 'wandb'), resume=True)

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
    if args.epochs <= args.warmup_epochs:
        args.warmup_epochs = args.epochs
        
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )

    # 코드북 사용률 추적을 위한 변수
    codebook_usage_tracker = torch.zeros(CODEBOOK_SIZE, dtype=torch.long, device=device)  # CODEBOOK_SIZE개 코드북
    codebook_log_interval = 10  # 100 iter마다 로깅

    # training loop
    X_text, Y_text = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    # early stopping 관련 변수
    patience = 10  # 개선 없을 때 몇 epoch 후 중단할지
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            # determine and set the learning rate for this iteration
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            
            X, Y_freq, Y_raw, input_chans, input_time, input_mask = batch
            # print(f"🔍 [Training] Batch loaded - X: {X.shape}, Y_freq: {Y_freq.shape}, Y_raw: {Y_raw.shape}")
            # print(f"🔍 [Training] input_chans: {input_chans.shape}, input_time: {input_time.shape}, input_mask: {input_mask.shape}")
            
            X = X.float().to(device, non_blocking=True)
            Y_freq = Y_freq.float().to(device, non_blocking=True)
            Y_raw = Y_raw.float().to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)

            with ctx:
                alpha = 2 / (1 + math.exp(-10 * iter_num / args.epochs / num_training_steps_per_epoch)) - 1
                loss, domain_loss, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, alpha)
                domain_loss2 = model(X_text)
                loss = (loss + domain_loss + domain_loss2) / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
            # 코드북 사용률 추적 (학습 중)
            with torch.no_grad():
                # 현재 배치의 코드북 인덱스 얻기
                mask = input_mask.unsqueeze(1).repeat(1, X.size(1), 1).unsqueeze(1)
                _, embed_ind, _, encoder_features = raw_model.VQ.encode(X, input_chans, input_time, mask)
                
                # 🔬 인코더 출력 유사성 진단 (매 10 iterations)
                if DEBUG_ENCODER and iter_num % 10 == 0:
                    # Encoder output을 flatten하고 normalize
                    enc_flat = encoder_features.reshape(-1, encoder_features.size(-1))  # (batch*tokens, dim)
                    enc_norm = torch.nn.functional.normalize(enc_flat, p=2, dim=-1)
                    
                    # 샘플링 (너무 크면)
                    n_samples = min(100, enc_norm.size(0))
                    if enc_norm.size(0) > n_samples:
                        idx = torch.randperm(enc_norm.size(0))[:n_samples]
                        enc_sample = enc_norm[idx]
                    else:
                        enc_sample = enc_norm
                    
                    # Pairwise cosine similarity
                    sim_matrix = torch.mm(enc_sample, enc_sample.t())
                    # 대각선 제외
                    mask_diag = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                    avg_sim = sim_matrix[mask_diag].mean().item()
                    
                    # Feature 표준편차 (다양성 지표)
                    feature_std = enc_flat.std(dim=0).mean().item()
                    
                    print(f"\n🔬 Encoder Diversity (iter {iter_num}):")
                    print(f"  Avg similarity: {avg_sim:.4f} (1.0=identical, 0.0=orthogonal)")
                    print(f"  Feature std: {feature_std:.4f} (0.0=collapsed)")
                
                # 사용된 인덱스 카운트 (flatten해서 모든 토큰 인덱스 추출)
                indices = embed_ind.flatten()
                for idx in indices:
                    codebook_usage_tracker[idx] += 1
            
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # clip the gradient
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num + 1) % args.log_interval == 0 and master_process:
                # Calculate progress
                total_batches = len(dataset_train) // args.batch_size
                progress_pct = (step + 1) / num_training_steps_per_epoch * 100
                
                # 코드북 사용률 계산 (최근 log_interval 동안의 사용)
                used_codes = (codebook_usage_tracker > 0).sum().item()
                codebook_usage_rate = used_codes / CODEBOOK_SIZE * 100
                
                # Top 10 가장 많이 사용된 코드
                top_k = min(10, used_codes) if used_codes > 0 else 0
                if top_k > 0:
                    top_codes = torch.topk(codebook_usage_tracker, k=top_k)
                    top_indices_str = ','.join([str(idx.item()) for idx in top_codes.indices[:5]])
                else:
                    top_indices_str = "None"
                
                print(f"[Epoch {epoch + 1}/{args.epochs}] "
                      f"[Batch {step + 1}/{num_training_steps_per_epoch} ({progress_pct:.1f}%)] "
                      f"[Iter {iter_num + 1}] "
                      f"Loss: {log['train/total_loss']:.4f} "
                      f"(freq: {log['train/rec_freq_loss']:.4f}, "
                      f"raw: {log['train/rec_raw_loss']:.4f}, "
                      f"quant: {log['train/quant_loss']:.4f}, "
                      f"domain: {log['train/domain_loss'] + domain_loss2.item():.4f}) "
                      f"LR: {lr:.2e} \n"
                      f"📊 Codebook: {used_codes}/{CODEBOOK_SIZE} ({codebook_usage_rate:.1f}%) Top5:[{top_indices_str}]")

                if args.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/total_loss": log['train/total_loss'],
                        "train/freq_loss": log['train/rec_freq_loss'],
                        "train/raw_loss": log['train/rec_raw_loss'],
                        "train/quant_loss": log['train/quant_loss'],
                        "train/domain_loss": log['train/domain_loss'] + domain_loss2.item(),
                        "lr": lr,
                        "codebook/used_codes": used_codes,
                        "codebook/usage_rate": codebook_usage_rate
                    })
                
                # 로그 출력 후 tracker 리셋 (최근 사용 패턴만 보기 위해)
                codebook_usage_tracker.zero_()
            
            X_text, Y_text = get_batch('train') # fetch the very first batch

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1
        
        # === validation loop ===
        model.eval()
        val_losses = []
        val_codebook_usage = torch.zeros(CODEBOOK_SIZE, dtype=torch.long, device=device)  # validation용 코드북 추적
        
        with torch.no_grad():
            for val_batch in data_loader_val:
                X, Y_freq, Y_raw, input_chans, input_time, input_mask = val_batch
                X = X.float().to(device, non_blocking=True)
                Y_freq = Y_freq.float().to(device, non_blocking=True)
                Y_raw = Y_raw.float().to(device, non_blocking=True)
                input_chans = input_chans.to(device, non_blocking=True)
                input_time = input_time.to(device, non_blocking=True)
                input_mask = input_mask.to(device, non_blocking=True)
                alpha = 0.0  # validation에서는 domain loss 영향 없음
                loss, domain_loss, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, alpha)
                val_losses.append(log['val/total_loss'])
                
                # Validation 코드북 사용률 추적
                mask = input_mask.unsqueeze(1).repeat(1, X.size(1), 1).unsqueeze(1)
                _, embed_ind, _, _ = raw_model.VQ.encode(X, input_chans, input_time, mask)
                indices = embed_ind.flatten()
                for idx in indices:
                    val_codebook_usage[idx] += 1
                    
        val_total_loss = float(torch.tensor(val_losses).mean().item())
        
        # Validation 코드북 사용률 계산
        val_used_codes = (val_codebook_usage > 0).sum().item()
        val_usage_rate = val_used_codes / CODEBOOK_SIZE * 100
        
        # 가장 많이 사용된 코드 Top 5
        top_codes = torch.topk(val_codebook_usage, k=min(5, val_used_codes))
        top_indices = top_codes.indices.cpu().tolist()
        top_counts = top_codes.values.cpu().tolist()
        
        model.train()

        if master_process:
            print(f"\n{'='*80}")
            print(f"[Epoch {epoch + 1}] Validation Results:")
            print(f"  Total Loss: {val_total_loss:.4f}")
            print(f"  📊 Codebook Usage: {val_used_codes}/{CODEBOOK_SIZE} ({val_usage_rate:.1f}%)")
            print(f"  Top 5 most used codes:")
            for i, (idx, count) in enumerate(zip(top_indices, top_counts)):
                print(f"    {i+1}. Code {idx}: {count} times")
            print(f"{'='*80}\n")
            
            if args.wandb_log:
                wandb.log({
                    "epoch": epoch + 1, 
                    "val/total_loss": val_total_loss,
                    "val/codebook_used": val_used_codes,
                    "val/codebook_usage_rate": val_usage_rate
                })

            # early stopping 체크
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                # 가장 좋은 모델 저장
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'encoder_args': encoder_args,
                    'decoder_args': decoder_args,
                    'iter_num': iter_num,
                    'epoch': epoch
                }
                print(f"[EarlyStopping] Best model saved at epoch {epoch + 1}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt_best.pt'))
            else:
                patience_counter += 1
                print(f"[EarlyStopping] Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"[EarlyStopping] Stop training at epoch {epoch + 1}")
                    break

            # 기존 checkpoint 저장
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--text_dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='NeuroLM')
    parser.add_argument('--wandb_runname', default='VQ')
    parser.add_argument('--wandb_api_key', type=str)
    # training args

    parser.add_argument('--gradient_accumulation_steps', default=DEFAULT_ACCUMULATION_STEPS, type=int)
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument('--text_batch_size', default=DEFAULT_TEXT_BATCH_SIZE, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    
    # VQ-VAE specific args
    parser.add_argument('--dead_code_threshold', type=float, default=1.0,
                        help='Dead code reset threshold. Codes with cluster_size below this will be reset. 0.0 disables.')

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
