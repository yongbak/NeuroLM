import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from pathlib import Path
from dataset import PickleLoader
from torch.utils.data import Dataset

# Time series augmentations
class TimeSeriesAugmentor:
    # Safe augmentations for malicious signals
    @staticmethod
    def gaussian_noise(signal, mean=0, std=0.01):
        if type(signal) == list:
            signal = np.array(signal)
        noise = np.random.normal(mean, std, signal.shape)
        return signal + noise

    @staticmethod
    def amplitude_scaling(signal, scale_min=0.9, scale_max=1.1):
        if type(signal) == list:
            signal = np.array(signal)
        scale = np.random.uniform(scale_min, scale_max)
        return signal * scale
    
class VAE(nn.Module):
    """Variational AutoEncoder for signal augmentation"""
    
    def __init__(self, 
                 input_dim=8000,      # Time series length
                 latent_dim=128,      # Latent space dimension
                 hidden_dims=[1024, 512, 256]):  # Encoder hidden layers
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = [latent_dim] + hidden_dims[::-1]
        for i in range(len(hidden_dims_rev) - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dims_rev[i], hidden_dims_rev[i+1]),
                nn.BatchNorm1d(hidden_dims_rev[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, noise_scale=1.0):
        """Reparameterization trick with optional noise scaling"""
        std = torch.exp(0.5 * logvar) * noise_scale
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to signal"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAugmentor:
    """VAE-based augmentation for signals"""
    
    def __init__(self,
                 pretrained_path=None,
                 input_dim=8000,
                 latent_dim=128,
                 hidden_dims=[512, 256, 128],
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.optimizer = None
        
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            self.load_model(pretrained_path)
        else:
            # Create new model
            self.model = VAE(input_dim, latent_dim, hidden_dims).to(device)
        
    def vae_loss(self, recon_x, x, mu, logvar, beta=0.001):
        """VAE loss = reconstruction loss + KL divergence"""
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(recon_x, x)
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kld, recon_loss, kld
    
    def train(self,
             train_dataloader,
             val_dataloader=None,
             epochs=100,
             lr=1e-3,
             beta=0.001,
             early_stopping_patience=5,
             min_delta=1e-4):
        """
        Train VAE on signal data from dataloader
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional, for early stopping)
            epochs: Maximum number of epochs
            lr: Learning rate
            beta: KL divergence weight
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Early stopping variables
        best_val_recon_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        print(f"\n{'='*60}")
        print(f"Starting VAE Training")
        print(f"Total epochs: {epochs}, Batches per epoch: {len(train_dataloader)}")
        print(f"Learning rate: {lr}, Beta (KL weight): {beta}")
        if val_dataloader:
            print(f"Early stopping: patience={early_stopping_patience}, min_delta={min_delta}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            total_recon = 0
            total_kld = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # batch[0] = list of X  (배치 내 모든 X값들)
                # batch[0].shape: (batch_size, block_size, sequence_unit)
                X = batch[0]

                # Flatten: (batch_size, block_size * sequence_unit)
                X_flat = X.reshape(X.shape[0], -1).to(self.device)
                
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(X_flat)
                loss, recon_loss, kld = self.vae_loss(recon_batch, X_flat, mu, logvar, beta)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld.item()
                
                # Print batch progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_dataloader)}] "
                          f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KLD: {kld.item():.4f}")
            
            # Print epoch summary
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_recon = total_recon / len(train_dataloader)
            avg_train_kld = total_kld / len(train_dataloader)
            
            print(f"\n[Epoch {epoch+1}/{epochs} Training Summary]")
            print(f"  Avg Loss: {avg_train_loss:.4f}, Avg Recon: {avg_train_recon:.4f}, Avg KLD: {avg_train_kld:.4f}")
            
            # Validation phase (if validation dataloader provided)
            if val_dataloader:
                self.model.eval()
                val_total_loss = 0
                val_total_recon = 0
                val_total_kld = 0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        X = batch[0]
                        X_flat = X.reshape(X.shape[0], -1).to(self.device)
                        
                        recon_batch, mu, logvar = self.model(X_flat)
                        loss, recon_loss, kld = self.vae_loss(recon_batch, X_flat, mu, logvar, beta)
                        
                        val_total_loss += loss.item()
                        val_total_recon += recon_loss.item()
                        val_total_kld += kld.item()
                
                avg_val_loss = val_total_loss / len(val_dataloader)
                avg_val_recon = val_total_recon / len(val_dataloader)
                avg_val_kld = val_total_kld / len(val_dataloader)
                
                print(f"[Epoch {epoch+1}/{epochs} Validation Summary]")
                print(f"  Avg Loss: {avg_val_loss:.4f}, Avg Recon: {avg_val_recon:.4f}, Avg KLD: {avg_val_kld:.4f}")
                
                # Early stopping check (based on validation reconstruction loss)
                if avg_val_recon < best_val_recon_loss - min_delta:
                    best_val_recon_loss = avg_val_recon
                    patience_counter = 0
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    print(f"  ✓ New best validation recon loss: {best_val_recon_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"  ✗ No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                # Check if early stopping triggered
                if patience_counter >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    print(f"Best validation recon loss: {best_val_recon_loss:.4f}")
                    print(f"{'='*60}\n")
                    
                    # Restore best model
                    if best_model_state:
                        self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                        print("Restored best model weights")
                    break
            
            print(f"{'-'*60}\n")
        
        # If training completed without early stopping, use best model if available
        if val_dataloader and best_model_state and patience_counter < early_stopping_patience:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            print(f"\nTraining completed. Restored best model with val recon loss: {best_val_recon_loss:.4f}")
    
    def test_and_generate(self, dataloader, num_samples_per_input=5, save_dir=None):
        """
        Test VAE and generate augmented samples
        
        Args:
            dataloader: DataLoader with test data
            num_samples_per_input: Number of augmented samples to generate per input
            save_dir: Directory to save generated samples (optional)
            
        Returns:
            dict with test metrics and generated samples
        """
        self.model.eval()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Testing VAE and Generating Augmented Samples")
        print(f"Samples per input: {num_samples_per_input}")
        print(f"{'='*60}\n")
        
        total_recon_loss = 0
        all_originals = []
        all_augmented = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                X = batch[0]
                X_flat = X.reshape(X.shape[0], -1).to(self.device)
                
                # Compute reconstruction loss
                recon_batch, mu, logvar = self.model(X_flat)
                recon_loss = nn.MSELoss()(recon_batch, X_flat)
                total_recon_loss += recon_loss.item()
                
                # Generate augmented samples for each input in batch
                for i in range(X_flat.shape[0]):
                    signal = X_flat[i:i+1]  # (1, input_dim)
                    mu_i, logvar_i = self.model.encode(signal)
                    
                    augmented_batch = []
                    for _ in range(num_samples_per_input):
                        z = self.model.reparameterize(mu_i, logvar_i)
                        aug = self.model.decode(z)
                        augmented_batch.append(aug.cpu().numpy())
                    
                    all_originals.append(signal.cpu().numpy())
                    all_augmented.append(augmented_batch)
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Processed batch [{batch_idx+1}/{len(dataloader)}], "
                          f"Recon Loss: {recon_loss.item():.4f}")
        
        avg_recon_loss = total_recon_loss / len(dataloader)
        
        print(f"\n[Test Summary]")
        print(f"  Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  Total original samples: {len(all_originals)}")
        print(f"  Total augmented samples: {len(all_originals) * num_samples_per_input}")
        print(f"{'='*60}\n")
        
        # Save if directory provided
        if save_dir:
            results = {
                'avg_recon_loss': avg_recon_loss,
                'originals': all_originals,
                'augmented': all_augmented
            }
            save_path = save_dir / 'test_results.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Test results saved to {save_path}")
        
        return {
            'avg_recon_loss': avg_recon_loss,
            'originals': all_originals,
            'augmented': all_augmented
        }
    
    def augment_single_sample(self, dataset_output, num_samples=1, noise_scale=0.9):
        """
        Generate augmented dataset outputs from a single dataset sample
        
        Args:
            dataset_output: tuple from dataset.__getitem__()
                           (X, Y_freq, Y_raw, input_chans, input_time, input_mask)
            num_samples: number of augmented samples to generate (default 1)
            noise_scale: scale factor for latent space variance (>1.0 for more diversity)
            
        Returns:
            return type: list[tuple]
            List of tuples, each containing:
            (X_aug, Y_freq_aug, Y_raw_aug, input_chans, input_time, input_mask)

            comment: [PickleLoader.__getitem__]
        """
        self.model.eval()
        
        # Unpack dataset output
        X, Y_freq, Y_raw, input_chans, input_time, input_mask = dataset_output
        
        # Flatten X for VAE
        original_shape = X.shape  # (block_size, sequence_unit)
        X_flat = X.reshape(1, -1)  # (1, block_size * sequence_unit)
        X_tensor = X_flat.to(self.device)
        
        augmented_outputs = []
        
        with torch.no_grad():
            # Encode to get mean and variance
            mu, logvar = self.model.encode(X_tensor)
            
            # Generate multiple augmented samples
            for _ in range(num_samples):
                # Augment in latent space by multipling noise
                z = self.model.reparameterize(mu, logvar, noise_scale=noise_scale)
                X_aug_flat = self.model.decode(z)
                
                # Reshape back to original
                X_aug = X_aug_flat.reshape(original_shape).cpu()
                
                # Recalculate Y_freq and Y_raw from augmented X
                # Determine actual data size (non-padded region)
                actual_tokens = input_mask.sum().int().item()
                
                Y_freq_aug = torch.zeros_like(Y_freq)
                Y_raw_aug = torch.zeros_like(Y_raw)
                
                if actual_tokens > 0:
                    # Extract actual data (non-padded part)
                    augmented_data = X_aug[:actual_tokens]
                    
                    # Recalculate Y_freq (FFT of augmented data)
                    x_fft = torch.fft.fft(augmented_data, dim=-1)
                    amplitude = torch.abs(x_fft)
                    # std_norm
                    mean = torch.mean(amplitude, dim=(0, 1), keepdim=True)
                    std = torch.std(amplitude, dim=(0, 1), keepdim=True)
                    amplitude = (amplitude - mean) / (std + 1e-8)
                    Y_freq_aug[:actual_tokens] = amplitude[:, :Y_freq.shape[1]]
                    
                    # Recalculate Y_raw (normalized augmented data)
                    mean = torch.mean(augmented_data, dim=(0, 1), keepdim=True)
                    std = torch.std(augmented_data, dim=(0, 1), keepdim=True)
                    Y_raw_aug[:actual_tokens] = (augmented_data - mean) / (std + 1e-8)
                
                # Create augmented output tuple
                augmented_output = (
                    X_aug,
                    Y_freq_aug,
                    Y_raw_aug,
                    input_chans,  # unchanged
                    input_time,   # unchanged
                    input_mask    # unchanged
                )
                augmented_outputs.append(augmented_output)
        
        return augmented_outputs
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reinitialize model with saved dimensions
        self.model = VAE(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dims=checkpoint.get('hidden_dims', [1024, 512, 256])  # 기존 체크포인트 호환
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


class AugmentedDataset(Dataset):
    """
    Dataset wrapper that augments each sample M times using trained VAE
    
    Given N original samples, produces M*N augmented samples
    """
    
    def __init__(self, 
                 original_dataset: Dataset,
                 vae_augmentor: VAEAugmentor,
                 num_augmentations_per_sample=5,
                 noise_scale=0.9,
                 include_original=False):
        """
        Args:
            original_dataset: Original dataset (e.g., PickleLoader)
            vae_augmentor: Trained VAEAugmentor instance
            num_augmentations_per_sample: Number of augmented versions per original sample (M)
            noise_scale: Scale factor for latent space noise
            include_original: If True, include original samples in the dataset
        """
        if vae_augmentor is None:
            raise ValueError("vae_augmentor must be provided")

        self.original_dataset = original_dataset
        self.vae_augmentor = vae_augmentor
        self.num_augmentations_per_sample = num_augmentations_per_sample
        self.noise_scale = noise_scale
        self.include_original = include_original
        
        # Calculate total dataset size
        self.original_size = len(original_dataset)
        
        # N original + M*N augmented = (M+1)*N total
        self.total_size = num_augmentations_per_sample * self.original_size
        if include_original:            
            self.total_size += self.original_size
        
        print(f"AugmentedDataset created:")
        print(f"  Original samples: {self.original_size}")
        print(f"  Augmentations per sample: {num_augmentations_per_sample}")
        print(f"  Include original: {include_original}")
        print(f"  Total samples: {self.total_size}")
    
    def __len__(self):
        """Return total number of samples (original + augmented)"""
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Get sample at index
        
        If include_original=True:
            - idx 0 to N-1: original samples
            - idx N to (M+1)*N-1: augmented samples
        
        If include_original=False:
            - idx 0 to M*N-1: augmented samples only
        """
        if self.include_original:
            # Original samples come first
            if idx < self.original_size:
                # Return original sample
                return self.original_dataset[idx]
            else:
                # Calculate which original sample to augment
                # OriginalA 1개, augmentationA N개, ... 순서로 등장
                aug_idx = idx - self.original_size
                original_idx = aug_idx // self.num_augmentations_per_sample
                
                # Get original sample
                original_sample = self.original_dataset[original_idx]
                
                # Generate augmented version
                augmented_samples = self.vae_augmentor.augment_single_sample(
                    original_sample, 
                    num_samples=1,
                    noise_scale=self.noise_scale
                )
                
                # num_samples=1로 세팅하고 항상 새로 반환하므로, 첫 번째 원소 반환
                return augmented_samples[0]
        else:
            # Only augmented samples
            # Calculate which original sample to augment
            original_idx = idx // self.num_augmentations_per_sample
            
            # Get original sample
            original_sample = self.original_dataset[original_idx]
            
            # Generate augmented version
            augmented_samples = self.vae_augmentor.augment_single_sample(
                original_sample,
                num_samples=1,
                noise_scale=self.noise_scale
            )
            
            return augmented_samples[0]


def create_augmented_dataloader(original_dataset,
                                vae_augmentor,
                                num_augmentations_per_sample=5,
                                noise_scale=0.9,
                                include_original=True,
                                batch_size=32,
                                shuffle=True,
                                num_workers=0):
    """
    Create a DataLoader with augmented dataset
    
    Args:
        original_dataset: Original dataset (N samples)
        vae_augmentor: Trained VAEAugmentor
        num_augmentations_per_sample: M augmentations per sample
        noise_scale: Latent space noise scale
        include_original: Include original samples
        batch_size: Batch size for DataLoader
        shuffle: Shuffle dataset
        num_workers: Number of worker processes
    
    Returns:
        DataLoader with M*N or (M+1)*N samples
    """
    from torch.utils.data import DataLoader
    
    augmented_dataset = AugmentedDataset(
        original_dataset=original_dataset,
        vae_augmentor=vae_augmentor,
        num_augmentations_per_sample=num_augmentations_per_sample,
        noise_scale=noise_scale,
        include_original=include_original
    )
    
    dataloader = DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    print(f"Created DataLoader with {len(augmented_dataset)} samples, batch_size={batch_size}")
    
    return dataloader


if __name__ == "__main__":
    import glob
    import os
    from pathlib import Path
    from torch.utils.data import ConcatDataset, DataLoader

    from constants import SAMPLING_RATE, NUM_OF_SAMPLES_PER_TOKEN, NUM_OF_TOTAL_TOKENS, NUM_WORKERS

    target = '*'
    model_name = {'*': "./vae_models/vae_augmentor.pt",
                  'b':"./vae_models/vae_augmentor_benign.pt",
                  'cc':"./vae_models/vae_augmentor_covert_channel.pt",
                  'm':"./vae_models/vae_augmentor_meltdown.pt",
                  's':"./vae_models/vae_augmentor_spectre.pt"}
    

    # 데이터 경로
    data_dir = "datasets/processed/PMD_samples"
    pattern = f"*_{target}_*.pkl"
    
    train_pkl_files = list((Path(data_dir)/"train").rglob(pattern))
    train_dataset = PickleLoader(train_pkl_files, block_size=NUM_OF_TOTAL_TOKENS, sampling_rate=SAMPLING_RATE, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)

    val_pkl_files = list((Path(data_dir)/"val").rglob(pattern))
    val_dataset = PickleLoader(val_pkl_files, block_size=NUM_OF_TOTAL_TOKENS, sampling_rate=SAMPLING_RATE, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    if len(train_pkl_files) == 0:
        print(f"No pickle files found in {data_dir}")
        exit(1)

    print(f"Found {len(train_pkl_files)} train files, {len(val_pkl_files)} val files")

    print("Loading data from PickleLoader...")
    
    # VAEAugmentor 생성
    # 첫 번째 배치로 input_dim 확인
    sample_batch = next(iter(train_dataloader))
    X_sample = sample_batch[0]  # X만 추출
    input_dim = X_sample.shape[1] * X_sample.shape[2]  # block_size * sequence_unit
    
    augmentor = VAEAugmentor(
        input_dim=input_dim,
        latent_dim=128,
        hidden_dims=[1024, 512, 256]
    )
    
    # KL 유지 필요
    print(f"Input dimension: {input_dim}")
    print("Training VAEAugmentor...")
    augmentor.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=100,
        lr=5e-4,
        beta=0.01,
        early_stopping_patience=5,
        min_delta=5e-4
    )
    
    # 모델 저장
    model_path = model_name[target]
    augmentor.save_model(model_path)
    print(f"Training complete! Model saved to {model_path}")
    
    # Test and generate augmented samples
    print("\n" + "="*60)
    print("Testing VAE and generating augmented samples...")
    print("="*60)
    
    # 모델이 완성된 이후에만 AugmentDataset 쓸 수 있음
    test_pkl_files = list((Path(data_dir)/"test").rglob(pattern))
    test_dataset = PickleLoader(test_pkl_files, block_size=NUM_OF_TOTAL_TOKENS, sampling_rate=SAMPLING_RATE, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN)
    test_dataloader = DataLoader(AugmentedDataset(test_dataset, augmentor, num_augmentations_per_sample=1, noise_scale=0.9, include_original=True))
    
    #test_augmented_dataset = AugmentedDataset(test_dataset, augmentor, num_augmentations_per_sample=1, noise_scale=0.9, include_original=False)
    #test_dataloader = DataLoader(ConcatDataset(test_dataset, test_augmented_dataset), batch_size=64, shuffle=True, num_workers=NUM_WORKERS)


    test_results = augmentor.test_and_generate(
        dataloader=test_dataloader,
        num_samples_per_input=5,
        save_dir="vae_test_results"
    )
    
    print(f"\nTest complete!")
    print(f"Generated {len(test_results['augmented'])} sets of augmented samples")
    
    # ============================================================
    # Example: Create augmented dataset with M augmentations per sample
    # ============================================================
    print("\n" + "="*60)
    print("Creating Augmented Dataset (M * N samples)")
    print("="*60)
    
    # Load trained augmentor
    trained_augmentor = VAEAugmentor(pretrained_path=model_path)
    
    # Create augmented dataset
    # N original samples → M*N augmented samples (or (M+1)*N if include_original=True)
    augmented_train_dataloader = create_augmented_dataloader(
        original_dataset=train_dataset,
        vae_augmentor=trained_augmentor,
        num_augmentations_per_sample=5,  # M=5: each sample gets 5 augmented versions
        noise_scale=0.9,
        include_original=True,  # Include original samples too
        batch_size=64,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    print(f"\nOriginal dataset size: {len(train_dataset)} samples")
    print(f"Augmented dataset size: {len(augmented_train_dataloader.dataset)} samples")
    print(f"Augmentation factor: {len(augmented_train_dataloader.dataset) / len(train_dataset):.1f}x")
    
    # Test the augmented dataloader
    print("\nTesting augmented dataloader...")
    sample_batch = next(iter(augmented_train_dataloader))
    print(f"Batch X shape: {sample_batch[0].shape}")
    print(f"Batch Y_freq shape: {sample_batch[1].shape}")
    print(f"Batch Y_raw shape: {sample_batch[2].shape}")
    
    print("Done!")
    