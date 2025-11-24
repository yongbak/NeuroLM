## Hyperparameters for training
NUM_WORKERS = 2  # ~10
DEFAULT_ACCUMULATION_STEPS = 8      # 1~
DEFAULT_BATCH_SIZE = 1              # ~16
DEFAULT_TEXT_BATCH_SIZE = 2         # ~64
DEFAULT_DTYPE = 'float16'           #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

NUM_OF_TOTAL_SAMPLES = 40000             # 1개 피클파일을 4만 샘플로 만듦
NUM_OF_SAMPLES_PER_TOKEN = 2000          # 1개 토큰을 만들기 위해 사용하는 샘플의 개수
NUM_OF_TOTAL_TOKENS = -(-NUM_OF_TOTAL_SAMPLES // NUM_OF_SAMPLES_PER_TOKEN)           # [정수 보장] 트랜스포머가 한 번에 입력받아 생산하는 토큰의 개수, time embedding과 관련이 있음. 

SAMPLING_RATE = 2000

OFFLINE = False

## Data Augmentation Parameters
GAUSSIAN_NOISE_MEAN = 0.0
GAUSSIAN_NOISE_STD = 0.05
AMPLITUDE_SCALING_MIN = 0.95
AMPLITUDE_SCALING_MAX = 1.05

