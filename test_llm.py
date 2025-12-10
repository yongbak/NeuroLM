from utils import load_vq_model, get_token_string
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. VQ 모델로 신호를 토큰화
vq_model = load_vq_model("./vq_output/checkpoints/VQ/ckpt_best.pt")
token_string = get_token_string(vq_model, "signal.csv", identifier="TOK")
# 결과: "<TOK_703> <TOK_266> <TOK_536> ..."

# 2. LLM 로드 (Qwen 0.6B)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.6B")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-0.6B")

# 3. 프롬프트 생성 (작은 모델용 간단한 버전)
prompt = create_simple_anomaly_prompt(token_string)

# 4. 추론
inputs = tokenizer(prompt, return_tensors="pt")
outputs = llm_model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)