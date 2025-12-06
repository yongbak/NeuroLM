"""
LLM Fine-tuning Script for SignalLM Token Sequences
Uses TRL SFTTrainer with double quantization (QLoRA)
"""

import os
import argparse
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional
import json
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

# Import from local files
from utils import load_vq_model, get_token_string
from constants import NUM_OF_SAMPLES_PER_TOKEN, NUM_OF_TOTAL_TOKENS, SAMPLING_RATE


class SignalLMTokenDataset(Dataset):
    """Dataset for converting signal files to token sequences for LLM training"""
    
    def __init__(
        self,
        signal_files: List[str],
        labels: List[int],
        vq_model,
        tokenizer,
        chunk_size: int = NUM_OF_TOTAL_TOKENS,
        sequence_unit: int = NUM_OF_SAMPLES_PER_TOKEN,
        device: str = 'cuda',
        token_prefix: str = "TOK",
        max_length: int = 2048,
    ):
        """
        Args:
            signal_files: List of paths to signal files (txt/csv)
            labels: List of labels corresponding to each signal file
            vq_model: Trained VQ model for tokenization
            tokenizer: HuggingFace tokenizer for LLM
            chunk_size: Number of tokens per chunk (default: NUM_OF_TOTAL_TOKENS)
            sequence_unit: Samples per token (default: NUM_OF_SAMPLES_PER_TOKEN)
            device: Device for VQ model inference
            token_prefix: Prefix for neural tokens (default: "TOK_")
            max_length: Maximum sequence length for LLM
        """
        self.signal_files = signal_files
        self.labels = labels
        self.vq_model = vq_model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sequence_unit = sequence_unit
        self.device = device
        self.token_prefix = token_prefix
        self.max_length = max_length
        
        assert len(signal_files) == len(labels), "Number of files must match number of labels"
        
    def __len__(self):
        return len(self.signal_files)
    
    def create_prompt(self, token_text: str, label: int) -> str:
        """Create training prompt from token sequence and label"""
        # Format: Instruction + Neural Tokens + Label
        prompt = f"""### Instruction:
Analyze the following neural signal tokens and predict the label.

### Neural Tokens:
{token_text}

### Label:
{label}"""
        return prompt
    
    def __getitem__(self, idx):
        """Get a single training example"""
        signal_file = self.signal_files[idx]
        label = self.labels[idx]

        # Extract token string from signal file
        token_string = get_token_string(self.vq_model, signal_file, self.sequence_unit, self.device)

        # Create prompt
        prompt = self.create_prompt(token_string, label)

        return {
            'text': prompt,
            'label': label,
            'num_tokens': len(token_string.split()),
            'file_path': signal_file
        }
    
    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset"""
        data = []
        print(f"Converting {len(self)} signal files to token sequences...")
        
        for idx in range(len(self)):
            if idx % 10 == 0:
                print(f"  Processing {idx}/{len(self)}...")
            item = self[idx]
            data.append({
                'text': item['text'],
                'label': item['label'],
            })
        
        return Dataset.from_list(data)


def load_model_and_tokenizer(
    model_name_or_path: str,
    is_offline: bool = False,
    use_flash_attention: bool = False,
):
    """
    Load LLM model with double quantization (QLoRA)
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        is_offline: Whether to load from local path
        use_flash_attention: Whether to use Flash Attention 2
    """
    print(f"Loading model from: {model_name_or_path}")
    print(f"Offline mode: {is_offline}")
    
    # BitsAndBytesConfig for double quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",              # Use NF4 quantization
        bnb_4bit_use_double_quant=True,         # Double quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=is_offline,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # Add neural tokens to vocabulary
    print("Adding neural tokens to vocabulary...")
    neural_tokens = [f"TOK_{i}" for i in range(1024)]  # CODEBOOK_SIZE = 1024
    num_added = tokenizer.add_tokens(neural_tokens)
    print(f"  Added {num_added} neural tokens")
    
    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    
    if is_offline:
        model_kwargs["local_files_only"] = True
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    
    # Resize token embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
):
    """Create LoRA configuration for PEFT"""
    
    # Default target modules (adjust based on your model architecture)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=r,                              # LoRA rank
        lora_alpha=lora_alpha,            # LoRA alpha
        target_modules=target_modules,    # Modules to apply LoRA
        lora_dropout=lora_dropout,        # Dropout probability
        bias="none",                      # Bias type
        task_type="CAUSAL_LM",            # Task type
    )
    
    return lora_config


def main():
    parser = argparse.ArgumentParser('LLM Fine-tuning for SignalLM')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or local path')
    parser.add_argument('--offline', action='store_true',
                        help='Load model from local path (offline mode)')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='Use Flash Attention 2')
    
    # VQ model arguments
    parser.add_argument('--vq_checkpoint', type=str, required=True,
                        help='Path to trained VQ model checkpoint')
    parser.add_argument('--vq_device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for VQ model inference')
    
    # Data arguments
    parser.add_argument('--train_files', type=str, required=True,
                        help='Path to file containing list of training signal files')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Path to file containing training labels')
    parser.add_argument('--eval_files', type=str, default=None,
                        help='Path to file containing list of eval signal files')
    parser.add_argument('--eval_labels', type=str, default=None,
                        help='Path to file containing eval labels')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./llm_output',
                        help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                        help='Warmup ratio')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Logging steps')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint steps')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*80)
    print("LLM Fine-tuning with SignalLM Tokens")
    print("="*80)
    
    # Load VQ model
    print("\n[1/6] Loading VQ model...")
    vq_model = load_vq_model(args.vq_checkpoint, args.vq_device, offline=True)
    vq_model.eval()
    
    # Load LLM model and tokenizer
    print("\n[2/6] Loading LLM model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        is_offline=args.offline,
        use_flash_attention=args.use_flash_attention,
    )
    
    # Apply LoRA
    print("\n[3/6] Applying LoRA...")
    lora_config = create_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load training data
    print("\n[4/6] Loading training data...")
    with open(args.train_files, 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    with open(args.train_labels, 'r') as f:
        train_labels = [int(line.strip()) for line in f if line.strip()]
    
    print(f"  Training files: {len(train_files)}")
    print(f"  Training labels: {len(train_labels)}")
    
    # Create training dataset
    train_dataset_wrapper = SignalLMTokenDataset(
        signal_files=train_files,
        labels=train_labels,
        vq_model=vq_model,
        tokenizer=tokenizer,
        device=args.vq_device,
        max_length=args.max_seq_length,
    )
    train_dataset = train_dataset_wrapper.to_hf_dataset()
    
    # Load evaluation data if provided
    eval_dataset = None
    if args.eval_files and args.eval_labels:
        print("\n[5/6] Loading evaluation data...")
        with open(args.eval_files, 'r') as f:
            eval_files = [line.strip() for line in f if line.strip()]
        with open(args.eval_labels, 'r') as f:
            eval_labels = [int(line.strip()) for line in f if line.strip()]
        
        print(f"  Eval files: {len(eval_files)}")
        print(f"  Eval labels: {len(eval_labels)}")
        
        eval_dataset_wrapper = SignalLMTokenDataset(
            signal_files=eval_files,
            labels=eval_labels,
            vq_model=vq_model,
            tokenizer=tokenizer,
            device=args.vq_device,
            max_length=args.max_seq_length,
        )
        eval_dataset = eval_dataset_wrapper.to_hf_dataset()
    else:
        print("\n[5/6] No evaluation data provided, skipping...")
    
    # Setup training arguments
    print("\n[6/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        bf16=True,                          # Use bfloat16 for training
        tf32=True,                          # Use TF32 for matmul
        optim="paged_adamw_32bit",          # Paged optimizer for QLoRA
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=["tensorboard"],
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )
    
    # Create SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,                      # Don't pack sequences
    )
    
    # Start training
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*80)
    print("Saving final model...")
    print("="*80)
    
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {final_model_path}")
    print(f"📊 Logs saved to: {os.path.join(args.output_dir, 'logs')}")
    
    # Print training summary
    if hasattr(trainer.state, 'log_history'):
        print("\n" + "="*80)
        print("Training Summary")
        print("="*80)
        final_log = trainer.state.log_history[-1]
        for key, value in final_log.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
