from utils import load_vq_model, get_token_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

def get_label_from_filename(filename):
    """Extract raw label character from filename (b/cc/m/s)"""
    parts = filename.split('-')
    if len(parts) > 1:      # is_augmented == True
        name = parts[1]
    else:
        name = parts[0]
    return name.split('_')[1]

def add_vq_tokens_to_tokenizer(tokenizer, vocab_size=1024):
    """
    Add VQ token vocabulary (<TOK_0> ~ <TOK_1024>) to tokenizer.json
    
    Args:
        tokenizer: HuggingFace tokenizer object
        vocab_size: Number of VQ codebook tokens (default: 1024)
    """
    
    tokenizer_path = tokenizer.vocab_files_names.get('tokenizer_file')
    if not tokenizer_path:
        # Try to find tokenizer.json in the model directory
        model_name = tokenizer.name_or_path
        potential_paths = [
            os.path.join(model_name, 'tokenizer.json'),
            'tokenizer.json'
        ]
        for path in potential_paths:
            if os.path.exists(path):
                tokenizer_path = path
                break
    
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        print(f"❌ tokenizer.json not found")
        return False
    
    # Load tokenizer.json
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    
    # Get current vocab
    current_vocab = tokenizer_json.get('model', {}).get('vocab', {})
    
    # Add VQ tokens
    for i in range(vocab_size):
        tok_str = f"<TOK_{i}>"
        if tok_str not in current_vocab:
            current_vocab[tok_str] = len(current_vocab)
    
    # Save updated tokenizer.json
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Added {vocab_size} VQ tokens to tokenizer.json")
    return True

def create_prompt(token_string, label):
    """
    Create prompt template for anomaly detection based on token distribution.
    
    Args:
        token_string: Token string from VQ model (e.g., "<TOK_257> <TOK_390> <TOK_912> ...")
    
    Returns:
        Prompt string for LLM
    """
    
    prompt = f"""신호 이상탐지 (Signal Anomaly Detection)

## 정상(Normal) 신호의 특징:

### 등장 횟수 기준 상위 20개 토큰:
1순위: <TOK_257> (21.3%)
2순위: <TOK_390> (19.5%)
3순위: <TOK_912> (14.6%)
4순위: <TOK_117> (7.6%)
5순위: <TOK_947> (6.3%)
6순위: <TOK_340> (6.0%)
7순위: <TOK_701> (5.5%)
8순위: <TOK_727> (4.9%) ⭐ 8~20순위가 핵심
9순위: <TOK_63> (3.9%)
10순위: <TOK_480> (3.4%)
11순위: <TOK_516> (2.2%)
12순위: <TOK_138> (1.4%)
13순위: <TOK_623> (0.9%)
14순위: <TOK_743> (0.6%)
15순위: <TOK_787> (0.6%)
16순위: <TOK_861> (0.5%)
17순위: <TOK_118> (0.4%)
18순위: <TOK_681> (0.3%)
19순위: <TOK_937> (0.3%)
20순위: <TOK_79> (0.2%)

**정상 신호에서 8~20순위(등장 횟수 기준)로 많이 나타나는 토큰들: <TOK_727>, <TOK_63>, <TOK_480>, <TOK_516>, <TOK_138>, <TOK_623>, <TOK_743>, <TOK_787>, <TOK_861>, <TOK_118>, <TOK_681>, <TOK_937>, <TOK_79>**

---

## 판정 기준:

입력된 신호의 토큰 등장 빈도를 기준으로 상위 20개를 추출하여 분석:

1. **정상 신호의 8~20순위 토큰들 확인**
   - <TOK_727>, <TOK_63>, <TOK_480>, <TOK_516>, <TOK_138>, <TOK_623>, <TOK_743>, <TOK_787>, <TOK_861>, <TOK_118>, <TOK_681>, <TOK_937>, <TOK_79>

2. **입력 신호에서도 같은 토큰들이 등장 빈도 상위 20위 내에 많이 포함되는지 확인**
   - 정상의 8~20순위 토큰들이 입력 신호에서도 상위 20위 내에 많이 나타남 → NORMAL
   - 정상의 8~20순위 토큰들이 입력 신호에서 상위 20위 밖으로 밀려남 → ABNORMAL

---

## 분석 대상 신호:

{token_string}

위 신호의 토큰 등장 빈도 기준 상위 20개를 추출하여:
1. 8~20순위(등장 횟수 기준)의 토큰들이 무엇인지 확인
2. 정상 신호의 8~20순위 토큰들(<TOK_727>, <TOK_63>, <TOK_480> 등)이 입력 신호의 상위 20위에 얼마나 포함되는지 비교
3. 정상 분포와의 유사도 판단

**최종 판정: [NORMAL / ABNORMAL]**
**신뢰도: [높음 / 중간 / 낮음]**
**이유: [8~20순위(등장 횟수 기준) 토큰의 일치도 분석 결과]**"""
    
    return prompt



# 1. VQ 모델로 신호를 토큰화
vq_model = load_vq_model("./vq_output/checkpoints/VQ/ckpt_29.pt")
token_string = get_token_string(vq_model, filename, identifier="TOK")
label = get_label_from_filename(filename)
# 결과: "<TOK_703> <TOK_266> <TOK_536> ..."

# 2. LLM 로드 (Qwen 0.6B)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.6B")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.6B")

# 3. 프롬프트 생성 (작은 모델용 간단한 버전)
prompt = create_prompt(token_string, )

# 4. 추론
inputs = tokenizer(prompt, return_tensors="pt")
outputs = llm_model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)