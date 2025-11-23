import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from pathlib import Path
from dataset import PickleLoader

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
                 hidden_dims=[512, 256, 128]):  # Encoder hidden layers
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
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
    """VAE-based augmentation for EEG signals"""
    
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
             dataloader,
             epochs=100,
             lr=1e-3,
             beta=0.001):
        """Train VAE on signal data from dataloader"""
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        print(f"\n{'='*60}")
        print(f"Starting VAE Training")
        print(f"Total epochs: {epochs}, Batches per epoch: {len(dataloader)}")
        print(f"Learning rate: {lr}, Beta (KL weight): {beta}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kld = 0
            
            for batch_idx, batch in enumerate(dataloader):
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
                    print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KLD: {kld.item():.4f}")
            
            # Print epoch summary
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon / len(dataloader)
            avg_kld = total_kld / len(dataloader)
            
            print(f"\n[Epoch {epoch+1}/{epochs} Summary]")
            print(f"  Avg Loss: {avg_loss:.4f}, Avg Recon: {avg_recon:.4f}, Avg KLD: {avg_kld:.4f}")
            print(f"{'-'*60}\n")
    
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
    
    def AugmentLoader(self, dataset_output, num_samples=1, noise_scale=0.9):
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
                # Sample from latent space with scaled variance
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
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reinitialize model with saved dimensions
        self.model = VAE(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import glob
    import os
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    # 데이터 경로
    data_dir = "datasets/processed/PMD_samples"
    pkl_files = list(Path(data_dir).rglob('*.pkl'))
    dataloader = PickleLoader(pkl_files, block_size=20, sampling_rate=2000, sequence_unit=2000)

    if len(pkl_files) == 0:
        print(f"No pickle files found in {data_dir}")
        exit(1)
    
    print(f"Found {len(pkl_files)} pickle files in {data_dir}")
    
    # PickleLoader로 데이터셋 생성
    dataset = PickleLoader(pkl_files)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Loading data from PickleLoader...")
    
    # VAEAugmentor 생성
    # 첫 번째 배치로 input_dim 확인
    sample_batch = next(iter(dataloader))
    X_sample = sample_batch[0]  # X만 추출
    input_dim = X_sample.shape[1] * X_sample.shape[2]  # block_size * sequence_unit
    
    augmentor = VAEAugmentor(
        input_dim=input_dim,
        latent_dim=128,
        hidden_dims=[1024, 512, 256]
    )
    
    print(f"Input dimension: {input_dim}")
    print("Training VAEAugmentor...")
    augmentor.train(
        dataloader=dataloader,
        epochs=100,
        lr=1e-4,
        beta=0.001
    )
    
    # 모델 저장
    model_path = "vae_augmentor.pt"
    augmentor.save_model(model_path)
    print(f"Training complete! Model saved to {model_path}")
    
    # Test and generate augmented samples
    print("\n" + "="*60)
    print("Testing VAE and generating augmented samples...")
    print("="*60)
    
    test_results = augmentor.test_and_generate(
        dataloader=dataloader,
        num_samples_per_input=5,
        save_dir="vae_test_results"
    )
    
    print(f"\nTest complete!")
    print(f"Generated {len(test_results['augmented'])} sets of augmented samples")
    print("Done!")
    