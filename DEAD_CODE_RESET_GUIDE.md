# Dead Code Reset 메커니즘 구현 가이드

## 🔍 문제 진단

### Codebook 사용률 급감 현상
```
Iteration 0:    100% (K-means 초기화 아티팩트)
Iteration 50:   88.7%
Iteration 100:  49.2%
Iteration 200:  23.8%
Validation:     11.7%
```

### 원인 분석

#### 1. **EMA의 구조적 문제**
```python
# EMA 업데이트 공식
cluster_size_new = 0.9 * cluster_size_old + 0.1 * bins_current

# 문제점:
# - 초반에 자주 사용된 코드는 cluster_size가 크게 유지됨
# - DECAY=0.9는 과거 90% + 현재 10% → 과거 편향이 매우 강함
# - 한 번 dominant해진 코드는 계속 dominant하게 유지됨
```

#### 2. **악순환 (Vicious Cycle)**
```
1. 특정 코드(예: Code 62)가 초기에 많이 사용됨
   ↓
2. EMA로 해당 코드의 임베딩이 강화됨
   ↓
3. 다양한 입력도 해당 코드에 매핑되기 시작
   ↓
4. 나머지 코드들은 사용 빈도 감소
   ↓
5. cluster_size가 0에 가까워짐 → "Dead Code"
   ↓
6. Dead code는 업데이트되지 않아 영원히 사용 안됨
```

#### 3. **왜 발생하는가?**
- **높은 DECAY (0.9)**: 과거 가중치가 너무 높음
- **EMA의 특성**: 과거 정보를 누적하여 smooth하게 업데이트
- **초기화 민감성**: K-means 초기화가 불균형하면 계속 불균형 유지
- **No Gradient on Codebook**: EMA 방식이라 gradient로 교정 불가

---

## 💡 해결책: Dead Code Reset

### 핵심 아이디어
사용되지 않는 코드(dead code)를 **현재 활성 샘플로 재초기화**하여 다시 경쟁에 참여시킴

### 구현 로직

```python
def reset_dead_codes(self, z_flattened, encoding_indices):
    """
    Dead code를 활성 샘플로 재초기화
    
    Args:
        z_flattened: 현재 배치의 인코더 출력 (N, D)
        encoding_indices: 현재 배치의 양자화 인덱스 (N,)
    """
    # 1. Dead code 찾기
    dead_codes = (self.cluster_size < self.dead_code_threshold).nonzero(as_tuple=True)[0]
    
    if len(dead_codes) == 0:
        return
    
    # 2. 가장 많이 사용된 코드의 샘플들 찾기
    bins = torch.bincount(encoding_indices, minlength=self.num_tokens)
    most_used_code = bins.argmax()
    active_samples_mask = (encoding_indices == most_used_code)
    active_samples = z_flattened[active_samples_mask]
    
    # 3. Dead code들을 랜덤 샘플로 재초기화
    n_dead = len(dead_codes)
    n_samples = len(active_samples)
    
    if n_samples >= n_dead:
        indices = torch.randperm(n_samples)[:n_dead]
    else:
        indices = torch.randint(0, n_samples, (n_dead,))
    
    reset_samples = active_samples[indices]
    reset_samples = l2norm(reset_samples)  # Normalize
    
    # 4. 임베딩 업데이트
    with torch.no_grad():
        self.embedding.weight.data[dead_codes] = reset_samples
        # Cluster size도 초기화 (완전 0이면 다시 dead가 됨)
        self.cluster_size.data[dead_codes] = self.dead_code_threshold + 1.0
    
    print(f"🔄 Reset {len(dead_codes)} dead codes")
```

### 핵심 설계 포인트

#### 1. **Dead Code 판정 기준**
```python
dead_codes = (self.cluster_size < dead_code_threshold).nonzero()

# Threshold 설정:
# - 0.0: 완전히 사용되지 않은 코드만 reset
# - 1.0: cluster_size < 1.0인 코드 reset (권장)
# - 10.0: 더 공격적으로 reset (사용률 매우 낮은 코드도 포함)
```

#### 2. **어떤 샘플로 재초기화?**
- **가장 많이 사용된 코드의 샘플들** 사용
- 이유: 해당 코드는 과도하게 사용되고 있으므로, 분할하여 diversity 향상

#### 3. **언제 Reset?**
```python
# EMA 업데이트 직후 실행
if self.training and self.embedding.update:
    # EMA update
    norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
    
    # Dead code reset (매 iteration마다)
    self.reset_dead_codes(z_flattened, encoding_indices)
```

#### 4. **Cluster Size 초기화**
```python
self.cluster_size.data[dead_codes] = self.dead_code_threshold + 1.0

# 왜 threshold + 1.0?
# - 0으로 초기화하면 다음 iteration에 바로 다시 dead로 판정될 수 있음
# - 약간의 "생존 버퍼"를 줘서 경쟁 기회 제공
```

---

## 🚀 사용 방법

### 1. Training 시작
```bash
# Dead code reset 활성화 (threshold=1.0)
python train_vq.py \
    --dead_code_threshold 1.0 \
    --dataset_dir /path/to/data \
    --batch_size 32 \
    --epochs 50

# Dead code reset 비활성화
python train_vq.py \
    --dead_code_threshold 0.0 \
    --dataset_dir /path/to/data
```

### 2. Threshold 튜닝 가이드

#### Conservative (보수적)
```bash
--dead_code_threshold 0.0   # 완전히 사용 안된 코드만
```
- 장점: 안정적, 기존 학습에 최소 영향
- 단점: 매우 낮은 사용률 코드는 살리지 못함

#### Balanced (균형) - **권장**
```bash
--dead_code_threshold 1.0   # cluster_size < 1.0
```
- 장점: 적절한 균형, 대부분의 경우 효과적
- 단점: 너무 자주 reset되면 학습 불안정 가능

#### Aggressive (공격적)
```bash
--dead_code_threshold 10.0  # cluster_size < 10.0
```
- 장점: 사용률 낮은 코드를 적극적으로 재활용
- 단점: 학습 불안정 위험, 수렴 느려질 수 있음

### 3. 모니터링

Training 로그에서 다음을 확인:
```
🔄 Reset 47 dead codes (cluster_size < 1.0)
   Dead codes: [5, 12, 23, 45, 67, 89, 102, 156, 203, 267, ...]
```

- **Reset 빈도**: 초반에는 자주, 학습 진행되면 감소 예상
- **Reset 개수**: 전체 코드북의 10% 이하가 이상적
- **패턴 확인**: 같은 코드가 반복적으로 reset되면 threshold 조정 필요

---

## 📊 예상 효과

### Before (Dead Code Reset 없이)
```
Epoch 1:  100% → 88% → 49% → 23% (급격한 collapse)
Validation: 11.7%
Top code dominance: Code 62 사용 3807회 (과도한 집중)
```

### After (Dead Code Reset 적용)
```
Epoch 1:  100% → 92% → 78% → 65% (안정적 유지)
Validation: 60%+ (기대)
Top code dominance: 더 균등한 분포
```

### 개선 지표
- ✅ **Codebook 사용률**: 23.8% → 60%+ (목표)
- ✅ **Validation gap**: 12% → 5% 이하 (train-val 일치)
- ✅ **Code 집중도**: 감소 (dominant code의 사용 빈도 낮아짐)
- ✅ **Diversity**: Encoder 출력 다양성 유지

---

## ⚙️ 추가 권장 사항

### 1. **DECAY 조정과 병행**
```python
# constants.py
DECAY = 0.75  # 0.9 → 0.75로 낮춤 (현재를 더 반영)

# Dead code reset과 함께 사용하면 시너지
# - DECAY 낮추면: 현재 데이터 반영 ↑
# - Dead code reset: 사용 안되는 코드 재활용 ↑
```

### 2. **BETA 조정**
```python
# constants.py
BETA = 0.1  # 0.25 → 0.1 (commitment loss 감소)

# Encoder에게 더 자유롭게 표현하도록 허용
```

### 3. **Encoder Diversity 모니터링**
```python
# constants.py
DEBUG_ENCODER = True  # 계속 켜두기

# 출력 해석:
# Avg similarity > 0.9: 인코더 문제, architecture 수정 필요
# Avg similarity < 0.7: 정상, quantizer 문제만 해결하면 됨
```

### 4. **주기적 Reset 고려**
현재는 매 iteration마다 reset하지만, 더 안정적으로 하려면:
```python
# norm_ema_quantizer.py에 추가 가능
if iter_num % reset_interval == 0:
    self.reset_dead_codes(...)
```

---

## 🔬 실험 체크리스트

### Phase 1: Baseline 확인
- [ ] Dead code reset 없이 학습 (현재 상태)
- [ ] Codebook 사용률 기록 (epoch별)
- [ ] Encoder diversity 기록

### Phase 2: Dead Code Reset 적용
- [ ] `--dead_code_threshold 1.0`으로 학습
- [ ] Reset 빈도 및 개수 모니터링
- [ ] Codebook 사용률 비교

### Phase 3: Hyperparameter 튜닝
- [ ] DECAY: 0.9 → 0.75 또는 0.7
- [ ] BETA: 0.25 → 0.1
- [ ] Threshold: 0.5, 1.0, 5.0 실험

### Phase 4: 성능 평가
- [ ] Reconstruction loss 확인
- [ ] Downstream task 성능 (있다면)
- [ ] Codebook 사용률 안정성

---

## 🎯 성공 기준

### Minimal Success
- Codebook 사용률 > 30% (epoch 1 끝)
- Validation 사용률 > 25%

### Target Success
- Codebook 사용률 > 50% (epoch 1 끝)
- Validation 사용률 > 45%
- Train-val gap < 10%

### Optimal Success
- Codebook 사용률 > 70%
- Validation 사용률 > 65%
- Top code 사용 빈도 < 2x average

---

## 🐛 Troubleshooting

### 문제: Reset이 너무 자주 발생
```
🔄 Reset 400+ dead codes (매 iteration)
```
**해결**: Threshold를 낮추기 (1.0 → 0.5 또는 0.0)

### 문제: Reset이 전혀 발생하지 않음
```
No dead codes found
```
**원인**: Threshold가 너무 낮음 (0.0)  
**해결**: Threshold 올리기 (0.0 → 1.0 또는 5.0)

### 문제: 학습이 불안정해짐
```
Loss가 튀거나 NaN 발생
```
**해결**: 
1. Threshold 낮추기 (공격성 줄이기)
2. Reset 주기 늘리기 (매 iteration → 매 10 iterations)
3. Learning rate 낮추기

### 문제: 여전히 collapse 발생
```
사용률이 계속 떨어짐
```
**진단**:
1. Encoder diversity 확인 (avg_similarity > 0.9?)
2. DECAY 너무 높은지 확인 (0.9 → 0.7로 낮추기)
3. Dead code reset 로그 확인 (실제로 작동하는지)

---

## 📚 참고 자료

### Dead Code Reset의 이론적 배경
1. **Vector Quantization Literature**
   - "Neural Discrete Representation Learning" (VQ-VAE 원논문)
   - Codebook collapse는 well-known problem

2. **비슷한 기법들**
   - **K-means restart**: Dead centroid를 랜덤 샘플로 재초기화
   - **EMA with momentum reset**: Momentum 주기적 초기화
   - **Gumbel-Softmax annealing**: Temperature scheduling

3. **우리 구현의 특징**
   - EMA 기반 VQ에 적용 (learnable VQ와 다름)
   - 가장 많이 사용된 코드의 샘플 재활용 (diversity 향상)
   - Threshold 기반 adaptive reset (학습 진행에 따라 자동 조정)

---

## 마무리

Dead code reset은 **EMA 기반 VQ-VAE의 codebook collapse를 방지하는 강력한 기법**입니다.

- ✅ 구현 간단 (50줄 내외)
- ✅ 학습에 큰 영향 없음 (안정적)
- ✅ Hyperparameter 튜닝 여지 많음
- ✅ 이론적 배경 탄탄

현재 상황 (11.7% validation 사용률)에서는 **필수적인 기법**이며, DECAY/BETA 조정과 병행하면 시너지 효과를 기대할 수 있습니다.

**지금 바로 실험해보세요!** 🚀
