from utils import load_vq_model, get_token_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re
import glob

def get_label_from_filename(filename):
    """Extract raw label character from filename (b/cc/m/s)"""
    parts = filename.split('-')
    if len(parts) > 1:      # is_augmented == True
        name = parts[1]
    else:
        name = parts[0]
    return name.split('_')[1]

def parse_label_from_response(response):
    """
    Parse NORMAL/ABNORMAL label from LLM response.
    
    Args:
        response: Full LLM response string
    
    Returns:
        'NORMAL' or 'ABNORMAL' or 'UNKNOWN' if parsing fails
    
    Examples:
        "Result: NORMAL" -> 'NORMAL'
        "Result: ABNORMAL" -> 'ABNORMAL'
        "비정상 신호입니다. Result: ABNORMAL" -> 'ABNORMAL'
    """
    
    # Strategy 1: Look for "Result: NORMAL" or "Result: ABNORMAL"
    match = re.search(r'Result:\s*(NORMAL|ABNORMAL)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Strategy 2: Look for standalone NORMAL/ABNORMAL (case-insensitive)
    # But prioritize if it appears at the end
    lines = response.strip().split('\n')
    for line in reversed(lines):  # Check from bottom up
        if re.search(r'\b(NORMAL|ABNORMAL)\b', line, re.IGNORECASE):
            match = re.search(r'\b(NORMAL|ABNORMAL)\b', line, re.IGNORECASE)
            return match.group(1).upper()
    
    # Strategy 3: Check entire response as fallback
    if re.search(r'\bNORMAL\b', response, re.IGNORECASE):
        return 'NORMAL'
    if re.search(r'\bABNORMAL\b', response, re.IGNORECASE):
        return 'ABNORMAL'
    
    # Failed to parse
    return 'UNKNOWN'

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

def create_prompt(token_string, normal_dist_info=None, abnormal_dist_info=None):
    """
    Create prompt for anomaly detection based on token distribution characteristics.
    
    분포의 "집중도"를 기반으로 판정:
    - NORMAL: 한두 개 토큰이 지배적 (뾰족한 분포)
    - ABNORMAL: 여러 토큰이 균등 분산 (평탄한 분포)
    
    Args:
        token_string: Token string from VQ model (e.g., "<TOK_776> <TOK_687> ...")
        normal_dist_info: Dict with keys like 'top_10_ratio', 'top_50_ratio', 'entropy'
        abnormal_dist_info: Same format as normal_dist_info
    
    Returns:
        Prompt string for LLM
    """
    
    # 기본값 설정 (analyze_tokens_unified.py 결과 기반)
    if normal_dist_info is None:
        normal_dist_info = {
            'top_1_token': 'Token 776',
            'top_1_ratio': 7.19,
            'top_10_ratio': 45.0,  # 상위 10개 누적 비율 (약)
            'top_50_ratio': 90.0,  # 상위 50개 누적 비율 (약)
            'description': '한두 개 토큰이 강하게 지배적 (뾰족한 분포)',
        }
    
    if abnormal_dist_info is None:
        abnormal_dist_info = {
            'top_1_token': 'Token 110',
            'top_1_ratio': 2.18,
            'top_10_ratio': 20.0,  # 상위 10개 누적 비율 (약)
            'top_50_ratio': 50.0,  # 상위 50개 누적 비율 (약)
            'description': '많은 토큰들이 거의 동등하게 분산 (평탄한 분포)',
        }
    
    prompt = f"""신호 이상탐지 - 토큰 분포 기반 분석 (Signal Anomaly Detection)

## 📊 분석 원리:

정상과 비정상 신호는 **토큰 분포의 형태**가 다릅니다:

### NORMAL 신호의 특징:
- **분포 형태**: 뾰족함 (Sharp distribution)
- **최상위 토큰**: {normal_dist_info['top_1_token']} ({normal_dist_info['top_1_ratio']:.2f}% 점유)
- **상위 10개 누적**: ~{normal_dist_info['top_10_ratio']:.0f}% (높은 집중도)
- **상위 50개 누적**: ~{normal_dist_info['top_50_ratio']:.0f}% 
- **의미**: {normal_dist_info['description']}

### ABNORMAL 신호의 특징:
- **분포 형태**: 평탄함 (Flat distribution)
- **최상위 토큰**: {abnormal_dist_info['top_1_token']} ({abnormal_dist_info['top_1_ratio']:.2f}% 점유)
- **상위 10개 누적**: ~{abnormal_dist_info['top_10_ratio']:.0f}% (낮은 집중도)
- **상위 50개 누적**: ~{abnormal_dist_info['top_50_ratio']:.0f}%
- **의미**: {abnormal_dist_info['description']}

---

## 🔍 판정 기준:

입력 신호의 토큰 분포를 분석하여:

1. **분포 집중도 확인**
   - 상위 5개 토큰이 전체의 몇 %를 차지하는가?
   - 상위 10개 토큰이 전체의 몇 %를 차지하는가?
   - NORMAL: 30~50% (집중)
   - ABNORMAL: 10~20% (분산)

2. **최상위 토큰 분석**
   - 최상위 토큰이 얼마나 지배적인가?
   - NORMAL: 최상위 토큰이 5% 이상 (뾰족함)
   - ABNORMAL: 최상위 토큰이 2~3% 정도 (평탄함)

3. **분포의 다양성**
   - 사용되는 고유 토큰의 수가 많은가?
   - 여러 토큰이 비슷한 빈도로 나타나는가?
   - NORMAL: 토큰이 한정적, 일부 지배적
   - ABNORMAL: 토큰이 다양함, 균등 분산

---

## 분석 대상 신호 (200 토큰):

{token_string}

---

## 📋 분석 작업:

위 신호의 토큰 분포를 계산하여:
1. 상위 5개, 10개, 20개 토큰의 누적 비율 계산
2. 최상위 토큰이 차지하는 비율
3. 분포의 집중도 (뾰족한가? 평탄한가?)
4. NORMAL 분포와 ABNORMAL 분포 중 어느 쪽에 더 가까운가?

## 🎯 최종 판정:

집중도가 높으면 (뾰족하면) → **Result: NORMAL**
집중도가 낮으면 (평탄하면) → **Result: ABNORMAL**

## 출력 예시:

정상 신호 예:
상위 5개 누적 비율: 35%
최상위 토큰: 8%
분포: 뾰족한 형태
Result: NORMAL

비정상 신호 예:
상위 5개 누적 비율: 12%
최상위 토큰: 2.5%
분포: 평탄한 형태
Result: ABNORMAL
"""
    
    return prompt



# ===== Configuration =====
VQ_CHECKPOINT = "C:\\Users\\myqkr\\Desktop\\SignalLM\\ckpt-19.pt"
DATA_DIR = "C:\\Users\\myqkr\\Desktop\\SignalLM\\pkl_data\\test"
LLM_MODEL = "Qwen/Qwen-0.6B"
DEVICE = "cpu"

# ===== Main Execution =====
if __name__ == "__main__":
    # 1. Load VQ model
    print("🔄 Loading VQ model...")
    vq_model = load_vq_model(VQ_CHECKPOINT, device=DEVICE, weights_only=False)
    print(f"✅ VQ model loaded from {VQ_CHECKPOINT}\n")
    
    # 2. Get all pkl files
    files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
    print(f"📂 Found {len(files)} files in {DATA_DIR}\n")
    
    if len(files) == 0:
        print("❌ No pkl files found!")
        exit(1)
    
    # 3. Load LLM model
    print("🔄 Loading LLM model...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    print(f"✅ LLM model loaded: {LLM_MODEL}\n")
    
    # 4. Statistics
    correct = 0
    total = 0
    results = []
    
    # 5. Process each file
    for idx, filename in enumerate(files, 1):
        print("="*80)
        print(f"📄 Processing [{idx}/{len(files)}]: {os.path.basename(filename)}")
        print("="*80)
        
        # Extract tokens
        token_string = get_token_string(vq_model, filename, identifier="TOK")
        label = get_label_from_filename(os.path.basename(filename))
        
        # Create conversation
        conversation = [
            # Round 1: 태스크 설명
            {
                "role": "user",
                "content": "아날로그 전자기 신호를 VQ-VAE를 사용해서 토큰화를 했다. 전체 20초 짜리 신호를 0.1초 단위로 나눠서, 하나의 토큰이 되도록 하여 총 200개의 토큰 시퀀스가 있다. 이 토큰 시퀀스를 분석해서 신호의 레이블-정상 혹은 비정상-을 제로샷으로 탐지해야 하는데, 그 방법을 이제부터 알려줄게."
            },
            {
                "role": "assistant",
                "content": "응. 신호 토큰 분석해서 정상 혹은 비정상으로 분류하겠습니다."
            },
            
            # Round 2: 판정 기준 설명
            {
                "role": "user",
                "content": "정상신호는 <TOK_257>, <TOK_390>, ...과 같은 토큰이 자주 나타나. 이게 자주 등장하는 토큰인데, 이 토큰은 정상 신호와 비정상 신호 모두에서 공통되게 자주 나타나는 토큰이야. 그런데 반대로, 자주 등장하지 않는 토큰은 정상신호에서만 나타나. 다시말해서, 정상신호에서 출현빈도가 낮은 토큰들이 등장한다면 그 토큰 시퀀스는 정상 신호일 가능성이 높아지고, 그 토큰들이 등장하지 않는다면 비정상 신호일 가능성이 높아지는거야."
            },
            {
                "role": "assistant",
                "content": "응 고마워. 그렇다면 이제 신호를 분석해볼까?"
            },
            
            # Round 3: 실제 분석 요청
            {
                "role": "user",
                "content": f"응, 이제 프롬프트를 전달할게.\n\n{create_prompt(token_string)}"
            }
        ]
        
        # Generate prompt with chat template
        prompt = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Inference
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse label
        predicted_label = parse_label_from_response(response)
        
        # Check correctness
        is_correct = (predicted_label == label.upper())
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            "filename": os.path.basename(filename),
            "true_label": label,
            "predicted_label": predicted_label,
            "correct": is_correct
        })
        
        # Print result
        print(f"🤖 Predicted: {predicted_label}")
        print(f"✅ True Label: {label}")
        print(f"{'✅ CORRECT!' if is_correct else '❌ WRONG'}")
        print()
    
    # 6. Print summary
    print("="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"Total Files: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {total - correct}")
    print(f"Accuracy: {correct/total*100:.2f}%")
    print("="*80)
    
    # 7. Print detailed results
    print("\n📋 Detailed Results:")
    print("-"*80)
    for result in results:
        status = "✅" if result["correct"] else "❌"
        print(f"{status} {result['filename']}: {result['true_label']} -> {result['predicted_label']}")
    print("="*80)

    from utils import load_vq_model, get_token_string
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os



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

def create_prompt(token_string, label=None, normal_tokens_set=None):
    """
    Create a prompt for LLM to analyze VQ-VAE tokens using token SET logic.
    
    기본 원리: 정상 신호에서는 특정 토큰 세트가 자주 나타나고,
    비정상 신호에서는 이 세트에 없는 다른 토큰들이 자주 나타남.
    -> "정상 분포에 없는 토큰이 보이면 비정상으로 판별"
    
    Args:
        token_string: Space-separated string of tokens (e.g., "<TOK_703> <TOK_266> ...")
        label: Optional label for debugging
        normal_tokens_set: Set of token IDs that appear in normal signals (default: ckpt-29 top 20)
    
    Returns:
        Prompt string for LLM
    """
    
    # Default normal token set from ckpt-29 analysis (top 20 tokens)
    # These are the tokens most frequently appearing in normal signals
    if normal_tokens_set is None:
        normal_tokens_set = {776, 687, 254, 1, 605, 582, 121, 789, 26, 117, 
                            207, 195, 58, 110, 535, 280, 47, 670, 819, 458}
    
    # Convert token IDs to token strings for readability in prompt
    normal_tokens_str = ", ".join([f"<TOK_{tok}>" for tok in sorted(normal_tokens_set)])
    
    prompt = f"""신호 이상탐지 (Signal Anomaly Detection) - Token SET 기반 분석

## 정상(Normal) 신호의 특징:

### 정상 신호에서 자주 나타나는 핵심 토큰 세트 (Token SET):
{normal_tokens_str}

**핵심 원리**: 
- 정상 신호는 위의 토큰들로 주로 구성됨 (33.5% 코드북 사용량)
- 비정상 신호는 이 세트에 없는 다른 토큰들이 많이 포함됨
- 정상 분포에 없는 토큰이 보이면 비정상으로 판별 가능

---

## 판정 기준 (Token SET 기반):

1. **입력 신호의 토큰 분석**
   - 신호를 토큰으로 변환
   - 나타나는 모든 토큰 목록 추출
   - 각 토큰의 등장 빈도 계산

2. **정상 토큰 세트와 비교**
   - 입력 신호의 토큰들이 정상 세트에 얼마나 포함되는지 확인
   - 정상 세트에 **없는** 새로운 토큰들 식별
   - 새로운 토큰들의 등장 빈도 확인

3. **최종 판정**
   - 입력 신호의 대부분 토큰이 정상 세트에 포함됨 → **NORMAL**
   - 입력 신호에서 정상 세트에 없는 새로운 토큰들이 많이/자주 나타남 → **ABNORMAL**

---

## 분석 대상 신호:

{token_string}

위 신호를 분석하여:

1. 신호에 나타나는 모든 토큰 추출
2. 정상 토큰 세트 확인: {{{normal_tokens_str}}}
3. 입력 신호의 토큰 중 정상 세트 포함도 계산 (%)
4. 정상 세트에 없는 이상 토큰 식별
5. "정상 분포에 없는 토큰이 보이면 비정상으로 판별" 원칙 적용

**최종 판정: [NORMAL / ABNORMAL]**
**신뢰도: [높음 / 중간 / 낮음]**
**정상 세트 포함도: [%]**
**이상 토큰 식별: [새로운 토큰들]**
**판정 근거: [정상 토큰 세트 포함도 및 이상 토큰 빈도 분석]**"""
    
    return prompt


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 1. VQ 모델로 신호를 토큰화
    vq_model = load_vq_model("./vq_output/checkpoints/VQ/ckpt_29.pt")
    token_string = get_token_string(vq_model, "signal.csv", identifier="TOK")
    # 결과: "<TOK_703> <TOK_266> <TOK_536> ..."

    # 2. LLM 로드 (Qwen 0.6B)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.6B")
    llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.6B")

    # 3. VQ 토큰을 tokenizer에 추가
    add_vq_tokens_to_tokenizer(tokenizer)

    # 4. 프롬프트 생성 (새로운 token SET 기반 로직)
    # 기본값 사용 (ckpt-29 분석 결과)
    prompt = create_prompt(token_string)
    
    # 또는 커스텀 normal_tokens_set 사용 가능:
    # custom_normal_set = {776, 687, 254, 1, 605, 582, 121, 789, ...}
    # prompt = create_prompt(token_string, normal_tokens_set=custom_normal_set)

    # 5. 추론
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)

'''
생각거리


자. 우리가 이제 알고있는건 아래와 같아.

NORMAL 분포에서 엄청 많이 쓰이고, ABNORMAL에서는 거의 안 쓰이는 토큰들
반대로, ABNORMAL 분포에서 많이 쓰이고, NORMAL에서는 거의 안 쓰이는 토큰들
하지만, 나는 200개의 토큰시퀀스를 전달하며 해당 시퀀스는 "쩡상"인지 "비정상"인지 물을 예정.
따라서,
NORMAL 분포에서 "엄청 많이 쓰이는" 토큰이 실제 200개 토큰시퀀스에서 "엄청 많이" 나오진 않아. 또, 그렇다고 normal 토큰이 abnormal에서 거의 안쓰이는건 또 아닌경우도 있어. "적게" 쓰일 뿐.

맞아, 그 포인트가 핵심이야.

200 토큰이면 개별 토큰이 “엄청 많이” 나올 수가 없고

NORMAL에서 많이 쓰이는 토큰도 ABNORMAL에서 0이 아니라 그냥 더 적게 나올 뿐이라
“이 토큰이 3% 이상이면 NORMAL” 같은 룰은 현실성이 거의 없어.

그래서 “절대 비율”이 아니라 “비율의 차이”를 여러 토큰에 걸쳐 조금씩 합산하는 방식으로 가야 해.
그게 사실상 Naive Bayes / log-odds 느낌이고, 짧은 시퀀스에 가장 잘 맞는 방법이야.

위 내용을 반영해서 prompt를 만들려면 어떻게 해야할까?
prompt 이상의 방법이 있을까?
'''