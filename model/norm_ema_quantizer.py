"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 
        if codebook_init_path == '':   
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
            
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)   

def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))

class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path='',
                dead_code_threshold=0.0):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold  # Dead code 판정 임계값 (0.0 = 사용안됨)
        
        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
            # Dead code tracking을 위한 버퍼 추가
            self.register_buffer('code_usage_count', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()
    
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)
            self.register_buffer('code_usage_count', torch.zeros(self.num_tokens))
            self.code_usage_count = self.code_usage_count.to(device)
    
    def reset_dead_codes(self, z_flattened, encoding_indices):
        """
        Dead code를 활성 샘플로 재초기화
        
        Args:
            z_flattened: 현재 배치의 인코더 출력 (N, D)
            encoding_indices: 현재 배치의 양자화 인덱스 (N,)
        """
        if not self.training or self.dead_code_threshold <= 0:
            return
        
        # Dead code 찾기 (cluster_size가 임계값 이하인 코드)
        dead_codes = (self.cluster_size < self.dead_code_threshold).nonzero(as_tuple=True)[0]
        
        if len(dead_codes) == 0:
            return
        
        # 현재 배치에서 많이 사용된 Top 10 코드들에서 샘플링 (다양성 증가)
        bins = torch.bincount(encoding_indices, minlength=self.num_tokens)
        top_k = min(10, (bins > 0).sum().item())  # 실제 사용된 코드 수와 10 중 작은 값
        
        if top_k == 0:
            return
        
        # Top K 코드들 선택
        top_codes = torch.topk(bins, k=top_k).indices
        
        # Top K 코드들에 할당된 모든 샘플들 수집
        active_samples_list = []
        for code_idx in top_codes:
            mask = (encoding_indices == code_idx)
            samples = z_flattened[mask]
            if len(samples) > 0:
                active_samples_list.append(samples)
        
        if len(active_samples_list) == 0:
            return
        
        # 모든 active 샘플들 합치기
        active_samples = torch.cat(active_samples_list, dim=0)
        
        # Dead code들을 랜덤 샘플로 재초기화
        n_dead = len(dead_codes)
        n_samples = len(active_samples)
        
        # 샘플 선택 (랜덤하게)
        if n_samples >= n_dead:
            indices = torch.randperm(n_samples, device=z_flattened.device)[:n_dead]
        else:
            indices = torch.randint(0, n_samples, (n_dead,), device=z_flattened.device)
        
        reset_samples = active_samples[indices]
        noise = torch.randn_like(reset_samples) * 0.1   # Noise 추가
        reset_samples = l2norm(reset_samples + noise)   # Noise 추가 후 normalize
        
        # Dead code들의 임베딩을 업데이트
        with torch.no_grad():
            self.embedding.weight.data[dead_codes] = reset_samples
            # Cluster size도 약간의 값으로 초기화 (완전 0이면 다시 dead가 됨)
            self.cluster_size.data[dead_codes] = self.dead_code_threshold + 1.0
            
        #print(f"🔄 Reset {len(dead_codes)} dead codes (cluster_size < {self.dead_code_threshold})")
        #print(f"   Dead codes: {dead_codes[:10].tolist()}{'...' if len(dead_codes) > 10 else ''}")

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        #z = rearrange(z, 'b c h w -> b h w c')
        z = l2norm(z) # by JWB, z: (b, n, c)
        z_flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flattened)
        
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'
        
        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        if self.training and self.embedding.update:
            #EMA cluster size

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            # self.embedding.cluster_size_ema_update(bins)
            # cluster_size = 0.9 * old_cluster_size + 0.1 * bins
            
            # def ema_inplace(moving_avg, new, decay):
            #     moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
                        
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)

            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
            
            # Dead code reset 적용
            self.reset_dead_codes(z_flattened, encoding_indices)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        #z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices
    
