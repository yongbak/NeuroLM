# batch/accumulation 늘리기, LR낮추고 warmup 늘리기

## Hyperparameters for training
NUM_WORKERS = 2  # ~10
DEFAULT_ACCUMULATION_STEPS = 8      # 1~
DEFAULT_BATCH_SIZE = 8             # ~16
DEFAULT_TEXT_BATCH_SIZE = 8         # ~64
DEFAULT_DTYPE = 'float16'           #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

NUM_OF_TOTAL_SAMPLES = 40000             # 1개 피클파일을 4만 샘플로 만듦
NUM_OF_SAMPLES_PER_TOKEN = 200          # 1개 토큰을 만들기 위해 사용하는 샘플의 개수
NUM_OF_TOTAL_TOKENS = -(-NUM_OF_TOTAL_SAMPLES // NUM_OF_SAMPLES_PER_TOKEN)           # [정수 보장] 트랜스포머가 한 번에 입력받아 생산하는 토큰의 개수, time embedding과 관련이 있음. 

SAMPLING_RATE = 2000

VAE_AUGMENT_FACTOR = 0

CODEBOOK_SIZE = 1024
DECAY = 0.9
BETA = 0.25
EMBEDDING_DIMENSION = 128

OFFLINE = False

'''
- 전체 배치의 모든 토큰들을 하나의 풀로 만듦
- 그 중 랜덤하게 100개 선택
- 100개 토큰들 간의 모든 쌍(pair)의 유사도 계산
- 100*99/2 = 4950개의 쌍

Avg similarity:
0.9 이상: 인코더 출력 거의 동일 → 문제!
0.7~0.9: 약간 유사 → 정상 범위 (같은 도메인이니까)
0.5 이하: 매우 다양함 → 이상적

Feature std:
0.01 이하: Collapsed → 문제!
0.05~0.2: 정상
0.3 이상: 매우 다양함


'''
DEBUG_ENCODER = True
DEBUG_DEADCODE_RESET = False

## Data Augmentation Parameters
GAUSSIAN_NOISE_MEAN = 0.0
GAUSSIAN_NOISE_STD = 0.1
AMPLITUDE_SCALING_MIN = 0.9
AMPLITUDE_SCALING_MAX = 1.1

