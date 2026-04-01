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

def load_examples(n=20):
    if not os.path.exists(DATA_PATH):
        return ""
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. '0:00' 타임스탬프인 '기본 사전' 데이터는 무조건 가져옴 (우선순위 1위)
    base_dict = df[df['timestamp'] == '0:00']
    
    # 2. 나머지는 랜덤으로 섞어서 가져옴
    other_examples = df[df['timestamp'] != '0:00'].sample(min(len(df[df['timestamp'] != '0:00']), n))
    
    # 두 데이터를 합침
    final_examples = pd.concat([base_dict, other_examples]) 
    
    example_str = "### 절대 준수해야 할 슬랭 매핑 가이드 (필수):\n"
    for _, row in final_examples.iterrows():
        example_str += f"'{row['text']}'의 느낌 -> {row['translated_text']}\n"
    return example_str

def translate_slang(user_input):
    """사용자 입력을 받아 찰진 인방 말투로 번역"""
    examples = load_examples(15) # 15개의 예시를 AI에게 보여줌 (Few-shot)
    
    prompt = f"""
너는 Kai Cenat의 전용 번역가야. 아래 제공된 [슬랭 매핑 가이드]는 한국 최고의 인방 전문가들이 작성한 것이니, 
기존의 사전적 의미는 무시하고 **무조건** 이 가이드에 적힌 뉘앙스로만 번역해.

[번역 지침]
1. 'Glazing'은 절대 '구라'가 아니다. '후빨', '빨아주기', '뇌절 찬양'으로 번역할 것.
2. 'Stand on business'는 '비즈니스'가 아니다. '가오 살리다', '할 말은 한다', '입 닫고 행동으로 보여주다'로 번역할 것.
3. 'Word to my mother'는 '엄마 걸고 진짜다', 'ㄹㅇ 팩트다'라는 뜻이다. 이상한 직역 하지 마.
4. 모든 문장은 한국의 디시인사이드, 펨코, 치지직 채팅창 느낌으로 '~함', '~임?', 'ㄹㅇ' 등을 써서 아주 무례하고 찰지게 만들어.

[슬랭 매핑 가이드]
{examples}

번역할 문장: "{user_input}"
결과(한국어 번역만 딱 출력해):
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