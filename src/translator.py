"""Kai-Talk: 미국 스트리머 슬랭 -> 한국 인터넷 방송 말투 번역기 (few-shot prompting)

모델을 학습시키지 않고, 데이터셋에서 입력과 관련된 예시를 골라 프롬프트에 넣는 방식입니다.
"""
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 프로젝트 루트의 .env 파일에서 GROQ_API_KEY를 읽어옵니다.

# ---------------------------------------------------------------- 설정
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
BASE_URL = "https://api.groq.com/openai/v1"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kai_slang_translated.csv")

DICT_MARKER = "0:00"  # '기본 사전' 행을 구분하는 값 (추후 별도 컬럼으로 정리 예정)
N_RELEVANT = 8        # 입력과 관련된 예시 최대 개수
N_RANDOM = 5          # 말투 참고용 랜덤 예시 개수

# 영어 기능어: 이 단어들이 겹친다고 관련 예시로 치지 않기 위한 목록
STOPWORDS = {
    "a", "an", "the", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "is", "am", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for", "and",
    "or", "but", "that", "this", "with", "my", "your", "so", "do", "did", "dont", "don't",
    "yo", "like", "just", "got", "gotta", "its", "it's", "im", "i'm",
}

SYSTEM_PROMPT = """너는 미국 스트리머 Kai Cenat의 방송 영어를 한국 인터넷 방송(치지직, 아프리카TV, 디시, 펨코) 채팅창 말투로 옮기는 번역가야.

규칙:
1. 먼저 문장의 실제 의미를 정확히 파악하고, 그 의미를 한국 인방 말투로 옮겨. 원문에 없는 내용을 덧붙이거나 의미를 바꾸지 마.
2. 아래 [참고 예시]에 같은 표현이 있으면 그 뉘앙스를 우선 따라.
3. 슬랭이 없는 평범한 문장도 같은 말투로 번역해.
4. 반말과 커뮤니티 말투(ㅋㅋ, ㄹㅇ, ~임 등)를 자연스럽게 써. 욕설은 원문의 강도에 맞을 때만 써.
5. 번역문 한 줄만 출력해. 설명, 따옴표, 원문은 쓰지 마."""

# ---------------------------------------------------------------- 내부 상태
_client = None
_df = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY가 없습니다. .env 파일에 키를 넣어주세요. (.env.example 참고)")
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def load_data():
    """데이터셋을 한 번만 읽어서 재사용합니다."""
    global _df
    if _df is None:
        df = pd.read_csv(DATA_PATH)
        _df = df.dropna(subset=["text", "translated_text"]).reset_index(drop=True)
    return _df


# ---------------------------------------------------------------- 예시 고르기
def _words(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def find_relevant(df, user_input, limit=N_RELEVANT):
    """입력 문장과 관련 있는 데이터 행을 점수순으로 최대 limit개 반환합니다."""
    lowered = user_input.lower()
    input_words = _words(lowered) - STOPWORDS
    scored = []
    for idx, row in df.iterrows():
        entry = str(row["text"]).lower().strip()
        score = 0
        if len(entry) >= 3 and entry in lowered:  # 사전 항목이 입력 안에 그대로 등장
            score += 10
        score += len(_words(entry) & input_words)  # 겹치는 단어 수
        if score > 0 and str(row.get("timestamp", "")) == DICT_MARKER:
            score += 2  # 기본 사전 행 가산점
        if score > 0:
            scored.append((score, idx))
    scored.sort(reverse=True)
    return df.loc[[idx for _, idx in scored[:limit]]]


def build_examples(user_input, mode="retrieval", seed=None):
    """mode: 'none'(예시 없음) / 'random'(랜덤만) / 'retrieval'(관련 예시 + 랜덤)"""
    if mode == "none":
        return ""
    df = load_data()
    if mode == "random":
        chosen = df.sample(min(len(df), N_RELEVANT + N_RANDOM), random_state=seed)
    elif mode == "retrieval":
        relevant = find_relevant(df, user_input)
        rest = df.drop(relevant.index)
        fillers = rest.sample(min(len(rest), N_RANDOM), random_state=seed)
        chosen = pd.concat([relevant, fillers])
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    lines = [f"'{r['text']}' -> {r['translated_text']}" for _, r in chosen.iterrows()]
    return "[참고 예시]\n" + "\n".join(lines)


# ---------------------------------------------------------------- 번역
def translate_slang(user_input, mode="retrieval", temperature=0.7, seed=None):
    """영어 문장을 한국 인방 말투로 번역합니다."""
    examples = build_examples(user_input, mode=mode, seed=seed)
    user_message = f"{examples}\n\n번역할 문장: \"{user_input}\"".strip()

    for attempt in range(3):  # 일시적인 오류(속도 제한 등)는 최대 3번 재시도
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip().strip('"“”')
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    print("🔥 Kai-Talk Slang Translator 가동 중... (종료: exit)")
    while True:
        try:
            user_text = input("\n[영어 슬랭 입력]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() == "exit":
            break
        print(f"[한국어 번역]: {translate_slang(user_text)}")