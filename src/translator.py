"""Kai-Talk: 미국 스트리머 슬랭 -> 한국어 번역기 (few-shot prompting)

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
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b-versatile은 2026-08-16 종료됨
BASE_URL = "https://api.groq.com/openai/v1"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kai_slang_translated.csv")

DICT_MARKER = "0:00"  # 구버전 CSV 호환용 (새 CSV는 kind 컬럼으로 구분)
N_RELEVANT = 8        # 입력과 관련된 예시 최대 개수
N_RANDOM = 5          # 말투 참고용 랜덤 예시 개수

# 톤별 설정: 프롬프트와, 예시로 쓸 CSV 컬럼이 함께 바뀝니다.
#   neutral : 뜻과 뉘앙스를 살린 자연스러운 구어체 (기본값)
#   inbang  : 한국 인터넷 방송/커뮤니티 말투
TONES = {
    "neutral": {
        "column": "translated_neutral",
        "prompt": """너는 미국 스트리머 Kai Cenat의 방송 영어를 한국어 자막으로 옮기는 번역가야.

규칙:
1. 먼저 문장의 실제 의미와 뉘앙스(감정, 강도, 장난스러운 정도)를 정확히 파악하고 그대로 옮겨. 원문에 없는 내용을 덧붙이거나 의미를 바꾸지 마.
2. 슬랭은 직역하지 말고, 한국어로 같은 뜻과 느낌이 나는 자연스러운 표현으로 바꿔. [참고 예시]에 같은 표현이 있으면 그 뜻과 뉘앙스를 따르되, 문장에 맞게 어미와 조사를 바꿔서 써.
3. 말투는 자연스러운 구어체 반말로 해. 인터넷 커뮤니티 은어, 초성 표현(ㄹㅇ, ㅇㅈ 등), 욕설과 비속어는 쓰지 마. 거친 감정은 '진짜, 완전, 너무' 같은 강조로 전달해.
4. 슬랭이 없는 평범한 문장은 평범하게 번역하고, 원문에 없는 강조나 감탄사를 붙이지 마.
5. 'bro'가 상대를 부르는 말이면 '브로'나 '야'로, 제3자를 가리키는 말이면 '쟤/걔/저 친구'로 옮겨. 부르는 말이 아닌데 '브로'를 그대로 넣지 마.
6. 번역문 한 줄만 출력해. 설명, 따옴표, 원문은 쓰지 마.""",
    },
    "inbang": {
        "column": "translated_text",
        "prompt": """너는 미국 스트리머 Kai Cenat의 방송 영어를 한국 인터넷 방송(치지직, 아프리카TV, 디시, 펨코) 채팅창 말투로 옮기는 번역가야.

규칙:
1. 먼저 문장의 실제 의미를 정확히 파악하고, 그 의미를 한국 인방 말투로 옮겨. 원문에 없는 내용을 덧붙이거나 의미를 바꾸지 마.
2. 아래 [참고 예시]에 같은 표현이 있으면 그 뉘앙스를 따르되, 단어를 그대로 끼워 넣지 말고 문장의 문법에 맞게 어미와 조사를 바꿔 자연스럽게 써.
3. 슬랭이 없는 평범한 문장도 같은 말투로 번역해.
4. 반말과 커뮤니티 말투(ㅋㅋ, ㄹㅇ, ~임 등)를 자연스럽게 써. 욕설은 원문의 강도에 맞을 때만 써.
5. 'bro'가 상대를 부르는 말이면 '브로'나 '야'로, 제3자를 가리키는 말이면 '쟤/걔/저 친구'로 옮겨. 부르는 말이 아닌데 '브로'를 그대로 넣지 마.
6. 번역문 한 줄만 출력해. 설명, 따옴표, 원문은 쓰지 마.""",
    },
}
DEFAULT_TONE = "neutral"

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
        df = pd.read_csv(DATA_PATH).dropna(subset=["text", "translated_text"])
        if "kind" not in df.columns:  # 구버전 CSV: 0:00 타임스탬프 = 사전 항목
            df["kind"] = df["timestamp"].astype(str).map(lambda t: "term" if t == DICT_MARKER else "sentence")
        if "translated_neutral" not in df.columns:  # 구버전 CSV 호환
            df["translated_neutral"] = ""
        if "note" not in df.columns:
            df["note"] = ""
        if "style_ok" not in df.columns:
            df["style_ok"] = "Y"
        df["note"] = df["note"].fillna("")
        df["translated_neutral"] = df["translated_neutral"].fillna("")
        df["style_ok"] = df["style_ok"].fillna("")
        _df = df.reset_index(drop=True)
    return _df


# ---------------------------------------------------------------- 예시 고르기
def _stem(word):
    """glazing / glazed / glaze 처럼 형태가 달라도 같은 어근으로 비교하기 위한 간단한 처리"""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word.rstrip("e")


def find_relevant(df, user_input, limit=N_RELEVANT):
    """입력에 등장하는 슬랭(사전 항목)과, 그 슬랭이 실제 문장에서 쓰인 예시를 찾아 반환합니다."""
    lowered = user_input.lower()
    input_stems = {_stem(w) for w in re.findall(r"[a-z']+", lowered)}
    term_hits, matched_stems = [], set()
    for idx, row in df[df["kind"] == "term"].iterrows():
        entry = str(row["text"]).lower().strip()
        if len(entry) < 3:  # W, L 같은 한 글자 항목은 오탐이 많아 제외
            continue
        if " " in entry:  # 여러 단어 표현은 문구 그대로 포함될 때만
            matched = entry in lowered
        else:
            matched = _stem(entry) in input_stems
        if matched:
            term_hits.append(idx)
            matched_stems.add(_stem(entry) if " " not in entry else entry)

    # 같은 슬랭이 들어간 문장 예시 (말투 참고용으로 쓸 수 있는 것만, 슬랭당 최대 2개)
    usage_hits = []
    for stem in matched_stems:
        count = 0
        for idx, row in df[(df["kind"] == "sentence") & (df["style_ok"] == "Y")].iterrows():
            text = str(row["text"]).lower()
            found = stem in text if " " in stem else stem in {_stem(w) for w in re.findall(r"[a-z']+", text)}
            if found and idx not in usage_hits:
                usage_hits.append(idx)
                count += 1
                if count == 2:
                    break
    return df.loc[(term_hits + usage_hits)[:limit]]


def _format_example(row, column):
    note = f" ({row['note']})" if row["note"] else ""
    return f"'{row['text']}'{note} -> {row[column]}"


def build_examples(user_input, mode="retrieval", seed=None, tone=DEFAULT_TONE):
    """mode: 'none'(예시 없음) / 'random'(랜덤만) / 'retrieval'(관련 예시 + 랜덤)"""
    if mode == "none":
        return ""
    column = TONES[tone]["column"]
    df = load_data()
    df = df[df[column].astype(str).str.strip() != ""]  # 선택한 톤의 번역이 있는 행만 예시로 사용
    style_pool = df[(df["kind"] == "sentence") & (df["style_ok"] == "Y") & (df["text"].str.len() <= 90)]  # 말투 참고용 문장 (너무 긴 것 제외)
    if mode == "random":
        chosen = style_pool.sample(min(len(style_pool), N_RELEVANT + N_RANDOM), random_state=seed)
    elif mode == "retrieval":
        relevant = find_relevant(df, user_input)
        rest = style_pool.drop(relevant.index, errors="ignore")
        fillers = rest.sample(min(len(rest), N_RANDOM), random_state=seed)
        chosen = pd.concat([relevant, fillers])
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    return "[참고 예시]\n" + "\n".join(_format_example(r, column) for _, r in chosen.iterrows())


# ---------------------------------------------------------------- 번역
def build_context_block(context):
    """직전 줄들(원문, 번역)을 상황 이해용 문맥으로 만듭니다. 번역 대상이 아님을 분명히 알려줍니다."""
    if not context:
        return ""
    lines = "\n".join(f"- {en} -> {ko}" for en, ko in context)
    return ("[앞 문맥: 상황을 이해하고 사람·제품 이름의 한국어 표기를 통일하는 데만 쓰고, 번역하지 마]\n"
            f"{lines}\n\n")


def translate_slang(user_input, mode="retrieval", temperature=0.7, seed=None, tone=DEFAULT_TONE, context=None):
    """영어 문장을 한국어로 번역합니다. tone: 'neutral'(기본) 또는 'inbang'(인방 말투)

    context: 직전 줄들의 [(영어 원문, 한국어 번역), ...] (선택). 영상 자막처럼 앞뒤 흐름이 중요할 때 사용합니다.
    """
    if tone not in TONES:
        raise ValueError(f"알 수 없는 tone: {tone} ({list(TONES)} 중 선택)")
    examples = build_examples(user_input, mode=mode, seed=seed, tone=tone)
    user_message = f"{examples}\n\n{build_context_block(context)}번역할 문장: \"{user_input}\"".strip()

    for attempt in range(3):  # 일시적인 오류(속도 제한 등)는 최대 3번 재시도
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": TONES[tone]["prompt"]},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip().strip('"“”')
            return text.replace("\u2011", "-")  # 줄바꿈 방지 하이픈(7‑Eleven)은 글꼴에 따라 깨지므로 일반 하이픈으로
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kai-Talk 슬랭 번역기")
    parser.add_argument("--tone", choices=list(TONES), default=DEFAULT_TONE,
                        help="neutral: 뜻과 뉘앙스 위주의 구어체 (기본) / inbang: 인터넷 방송 말투")
    args = parser.parse_args()

    print(f"🔥 Kai-Talk Slang Translator 가동 중... (톤: {args.tone}, 종료: exit)")
    while True:
        try:
            user_text = input("\n[영어 슬랭 입력]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() == "exit":
            break
        print(f"[한국어 번역]: {translate_slang(user_text, tone=args.tone)}")
