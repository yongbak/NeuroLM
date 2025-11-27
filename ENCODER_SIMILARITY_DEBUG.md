# Encoder Similarity 진단 및 대응 가이드

## 🔬 현재 측정 방법 요약

### 무엇을 측정하는가?
**서로 다른 토큰들 간의 코사인 유사도 평균**

```python
# 계산 과정:
1. encoder_features: (batch=32, tokens=200, dim=768)
2. Flatten → (6400, 768)  # 6400개의 서로 다른 토큰
3. L2 Normalize → 단위 벡터로 변환
4. Random sampling → 100개 토큰 선택
5. Pairwise similarity → 100x100 행렬
6. 대각선 제외 평균 → avg_similarity
```

### 비교 대상
- **토큰 A**: 샘플1, 위치10, 시간구간 2000~2200
- **토큰 B**: 샘플15, 위치150, 시간구간 30000~30200  
- **토큰 C**: 샘플28, 위치77, 시간구간 15400~15600

→ **완전히 다른 샘플, 다른 위치, 다른 시간의 토큰들끼리 비교**

---

## 🚨 높은 Similarity의 의미

### Similarity > 0.95
```
🔴 CRITICAL: 인코더 출력 붕괴 (Encoder Collapse)

원인:
- Encoder가 입력과 무관하게 거의 동일한 벡터 출력
- Weight initialization 문제
- Gradient vanishing/exploding
- 학습률이 너무 높거나 낮음
- Batch normalization 문제

결과:
- 모든 입력이 같은 코드북으로 매핑됨
- Codebook collapse의 근본 원인
- Reconstruction 불가능
```

### Similarity 0.85~0.95
```
🟡 WARNING: 다양성 부족

원인:
- Encoder capacity 부족 (layer/dim 너무 작음)
- Overfitting to dominant patterns
- Data augmentation 부족
- Position encoding 문제

조치:
- DECAY 낮추기 (0.9 → 0.7)
- Dropout 추가/증가
- Data augmentation 강화
```

### Similarity 0.65~0.85
```
🟢 NORMAL: 정상 범위

이유:
- 같은 도메인(EEG)이므로 어느정도 유사성 자연스러움
- 뇌파는 특정 패턴(alpha, beta wave 등) 반복
- 건강한 다양성 유지
```

### Similarity < 0.65
```
🟢 EXCELLENT: 매우 다양함

- 이상적인 상태
- Encoder가 입력의 미세한 차이도 잘 구분
- Codebook 활용도 높을 것으로 기대
```

---

## 🔧 문제 해결 방법

### 1단계: 원인 진단

```bash
# Training 로그 확인
grep "Encoder Diversity" train.log | tail -20

# 패턴 분석:
# - 갑자기 올라감: 학습 중 문제 발생 (gradient explosion?)
# - 처음부터 높음: Initialization 문제
# - 서서히 올라감: Overfitting or collapse 진행 중
```

### 2단계: Feature Std 함께 확인

```python
# Feature std가 함께 떨어지면 확실한 collapse
if avg_similarity > 0.9 and feature_std < 0.01:
    print("🔴 확실한 Encoder Collapse!")
    print("   → Encoder 재초기화 또는 architecture 변경 필요")
```

### 3단계: 즉각 대응

#### Option A: Encoder Initialization 재설정
```python
# model/model_neural_transformer.py에서

class NeuralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ...
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 더 작은 std로 초기화
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 0.02 → 0.01
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
```

#### Option B: Dropout 추가
```python
# encoder_config에 dropout 설정
encoder_args = dict(
    n_layer=8,
    n_head=8,
    n_embd=768,
    dropout=0.2,  # 0.0 → 0.2로 증가
    bias=False,
)
```

#### Option C: Learning Rate 조정
```bash
# 너무 높으면 발산, 너무 낮으면 collapse
python train_vq.py \
    --learning_rate 1e-5 \  # 5e-5 → 1e-5로 낮춤
    --warmup_epochs 20      # 10 → 20으로 증가
```

#### Option D: Gradient Clipping 강화
```bash
python train_vq.py \
    --grad_clip 1.0  # 0.0 → 1.0
```

#### Option E: Batch Size 조정
```bash
# 너무 작은 batch size는 불안정
python train_vq.py \
    --batch_size 32      # 4 → 32
    --gradient_accumulation_steps 4
```

---

## 📊 실시간 모니터링

### Training 중 확인 사항

```python
# 매 10 iterations마다 출력:
🔬 Encoder Diversity (iter 50):
  Avg similarity: 0.7234 (1.0=identical, 0.0=orthogonal)
  Feature std: 0.1456 (0.0=collapsed)

# 정상 패턴:
# iter 10:  0.75, std 0.14
# iter 50:  0.72, std 0.15
# iter 100: 0.69, std 0.16  → 점점 다양해짐 (good!)

# 문제 패턴:
# iter 10:  0.75, std 0.14
# iter 50:  0.85, std 0.08
# iter 100: 0.95, std 0.02  → 점점 붕괴됨 (bad!)
```

### 즉시 중단 기준
```python
if avg_similarity > 0.95 and feature_std < 0.01:
    print("🛑 STOP TRAINING!")
    print("   Encoder has collapsed. Restart with different hyperparameters.")
    # Training 중단하고 설정 변경
```

---

## 🧪 디버깅 코드

현재 상황을 더 자세히 파악하려면:

```python
# train_vq.py에 임시로 추가:
if DEBUG_ENCODER and iter_num % 10 == 0:
    # 기존 코드...
    
    # 추가 진단:
    # 1. 배치별 similarity
    batch_sims = []
    for b in range(encoder_features.size(0)):
        batch_tokens = encoder_features[b]  # (tokens, dim)
        batch_norm = F.normalize(batch_tokens, p=2, dim=-1)
        batch_sim = torch.mm(batch_norm, batch_norm.t())
        mask = ~torch.eye(batch_sim.size(0), dtype=torch.bool, device=batch_sim.device)
        batch_sims.append(batch_sim[mask].mean().item())
    
    print(f"  Per-sample similarity: {torch.tensor(batch_sims).mean():.4f} ± {torch.tensor(batch_sims).std():.4f}")
    
    # 2. 첫번째와 마지막 토큰의 similarity
    first_tokens = encoder_features[:, 0, :]  # (batch, dim)
    last_tokens = encoder_features[:, -1, :]   # (batch, dim)
    first_norm = F.normalize(first_tokens, p=2, dim=-1)
    last_norm = F.normalize(last_tokens, p=2, dim=-1)
    positional_sim = (first_norm * last_norm).sum(dim=-1).mean().item()
    print(f"  First-Last token similarity: {positional_sim:.4f}")
    
    # 3. 개별 토큰의 norm 확인
    token_norms = encoder_features.norm(dim=-1).mean().item()
    print(f"  Avg token norm: {token_norms:.4f}")
```

---

## 🎯 Target Metrics

### 건강한 Encoder의 지표
```
Avg similarity:  0.60 ~ 0.80
Feature std:     0.05 ~ 0.20
Token norm:      5.0 ~ 15.0 (normalize 전)
Codebook usage:  > 60%
```

### Collapse 징후
```
Avg similarity:  > 0.90
Feature std:     < 0.02
Token norm:      매우 크거나 작음 (< 1.0 or > 100)
Codebook usage:  < 20%
```

---

## 💡 Similarity가 갑자기 높아진 경우

### 즉시 체크리스트:

1. **Learning rate 확인**
   ```bash
   # Warmup 끝났는지 확인
   # iter_num과 warmup_steps 비교
   ```

2. **Gradient norm 확인**
   ```bash
   # Loss가 NaN이거나 infinity인지
   # Gradient explosion 가능성
   ```

3. **최근 변경사항 체크**
   ```bash
   # Dead code reset 적용 후인지?
   # DECAY/BETA 변경했는지?
   # Checkpoint에서 resume했는지?
   ```

4. **Data 확인**
   ```bash
   # 혹시 같은 배치가 반복되는지
   # Data augmentation이 꺼졌는지
   ```

---

## 📝 요약

**Avg Similarity = 랜덤하게 선택한 100개 토큰들 간의 평균 코사인 유사도**

- **비교 대상**: 서로 다른 샘플, 다른 위치의 토큰들
- **정상 범위**: 0.65 ~ 0.85
- **문제 징후**: > 0.90 (특히 feature_std < 0.02일 때)
- **대응**: Learning rate 낮추기, Dropout 추가, Initialization 재검토

**갑자기 높아졌다면**: 학습 중 문제 발생 → 즉시 원인 파악 필요!
