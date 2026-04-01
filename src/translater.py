import os
import pandas as pd
from openai import OpenAI

# 1. 설정
# 본인의 OpenAI API 키를 여기에 넣으세요 (환경 변수 사용 권장)
# os.environ["OPENAI_API_KEY"] = "sk-..." 
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", # Groq 서버로 연결
    api_key=os.getenv("OPENAI_API_KEY") 
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kai_slang_translated.csv")

def load_examples(n=10):
    """데이터셋에서 n개의 예시를 무작위로 추출하여 AI에게 학습시킴"""
    if not os.path.exists(DATA_PATH):
        return ""
    
    df = pd.read_csv(DATA_PATH)
    # 데이터가 부족하면 있는 만큼만 가져옴
    sample_size = min(len(df), n)
    samples = df.sample(sample_size)
    
    example_str = "참고할 번역 예시:\n"
    for _, row in samples.iterrows():
        example_str += f"영어: {row['text']} -> 한국어: {row['translated_text']}\n"
    return example_str

def translate_slang(user_input):
    """사용자 입력을 받아 찰진 인방 말투로 번역"""
    examples = load_examples(15) # 15개의 예시를 AI에게 보여줌 (Few-shot)
    
    prompt = f"""
너는 Kai Cenat 같은 미국 스트리머의 슬랭을 한국의 20대 남초 커뮤니티(디시, 펨코) 및 
인터넷 방송(치지직, 아프리카) 시청자들이 쓰는 '찰진 말투'로 번역하는 전문가야.

[번역 규칙]
1. 말투는 반드시 '~함', '~함?', 'ㄴㄴ', 'ㄹㅇ', '임' 등 간결하고 날카로운 인방 말투를 사용할 것.
2. 'Cooked', 'Rizz', 'Motion', 'Glaze', 'Cap', 'Deadass' 등의 키워드를 문맥에 맞게 '나락', '폼', '빨다', '구라' 등으로 의역할 것.
3. 절대 정중하거나 딱딱하게 번역하지 말고, 실제 친구끼리 대화하거나 채팅창에서 드립 치는 느낌을 살릴 것.

{examples}

번역할 문장: "{user_input}"
결과:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Groq에서 제공하는 고성능 모델
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("🔥 Kai-Talk Slang Translator 가동 중...")
    print("(종료하려면 'exit' 입력)")
    
    while True:
        user_text = input("\n[영어 슬랭 입력]: ")
        if user_text.lower() == 'exit':
            break
            
        result = translate_slang(user_text)
        print(f"[찰진 한국어 번역]: {result}")