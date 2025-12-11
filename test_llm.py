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

def create_prompt(token_string):
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

## 정상 및 비정상 판정 기준:

입력된 신호의 토큰 등장 빈도를 기준으로 상위 20개를 추출하여 분석:

1. **정상 신호의 8~20순위 토큰들 확인**
   - <TOK_727>, <TOK_63>, <TOK_480>, <TOK_516>, <TOK_138>, <TOK_623>, <TOK_743>, <TOK_787>, <TOK_861>, <TOK_118>, <TOK_681>, <TOK_937>, <TOK_79>

2. **입력 신호에서도 같은 토큰들이 등장 빈도 상위 20위 내에 많이 포함되는지 확인**
   - 정상의 8~20순위 토큰들이 입력 신호에서도 상위 20위 내에 많이 나타남 → NORMAL
   - 정상의 8~20순위 토큰들이 입력 신호에서 상위 20위 밖으로 밀려남 → ABNORMAL

---

## 분석 대상 신호:
{token_string}

## 출력 포맷
Result: [NORMAL / ABNORMAL]

## 출력 예시
정상인 경우
Result: NORMAL

비정상인 경우
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