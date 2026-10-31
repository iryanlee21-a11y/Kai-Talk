"""Kai-Talk 파이프라인: 유튜브 링크 -> 영어 문장 추출(miner) -> 한국어 번역(translator) -> CSV + 자막(.srt)

사용법 (프로젝트 루트에서):
    python src/pipeline.py "https://youtu.be/VIDEO_ID"
    python src/pipeline.py "https://youtu.be/VIDEO_ID" --tone inbang --model medium
    python src/pipeline.py "https://youtu.be/VIDEO_ID" --limit 20     # 앞 20줄만 먼저 시험

결과 (data/output/):
    <video_id>_<tone>.csv  : 시간, 원문, 번역 (중간에 끊겨도 이어서 번역)
    <video_id>_<tone>.srt  : 자막 파일 (VLC, 유튜브 자막 업로드 등에 사용)
"""
import argparse
import os
import time

import pandas as pd

import miner
import translator

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output")
MAX_CAPTION_SEC = 6.0   # 자막 한 줄을 최대 이만큼만 보여줌
MIN_CAPTION_SEC = 1.2
CONTEXT_GAP_SEC = 15.0  # 이 시간보다 오래 전 줄은 문맥으로 쓰지 않음
CHECK_CONF = 0.6        # 이보다 낮은 인식 신뢰도
CHECK_MIN_WORDS = 3     # 단, 짧은 감탄사("What?")는 정확해도 신뢰도가 낮게 나오므로 이 단어 수 이상일 때만 표시


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(df, path):
    """번역된 줄만 골라 자막으로 저장합니다. 끝 시간은 다음 문장 시작 시간(최대 6초)으로 정합니다."""
    rows = df[df["translation"].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)
    starts = df["start_sec"].tolist()
    lines = []
    for n, row in rows.iterrows():
        nxt = [s for s in starts if s > row["start_sec"]]
        end = min(nxt[0] if nxt else row["start_sec"] + MAX_CAPTION_SEC, row["start_sec"] + MAX_CAPTION_SEC)
        end = max(end, row["start_sec"] + MIN_CAPTION_SEC)
        lines.append(f"{n + 1}\n{srt_time(row['start_sec'])} --> {srt_time(end)}\n{row['translation']}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def needs_check(text, conf):
    """인식이 불확실해 보이는 줄인지: 단어가 충분히 많은데 신뢰도가 낮은 경우"""
    try:
        return len(str(text).split()) >= CHECK_MIN_WORDS and float(conf) < CHECK_CONF
    except (TypeError, ValueError):
        return False


def get_context(df, idx, n):
    """번역할 줄(idx) 바로 앞의 최근 n줄 중, 15초 이내이고 이미 번역된 줄을 (원문, 번역) 쌍으로 돌려줍니다."""
    pos = df.index.get_loc(idx)
    prev = df.iloc[max(0, pos - n):pos]
    prev = prev[(df.at[idx, "start_sec"] - prev["start_sec"] <= CONTEXT_GAP_SEC) & (prev["translation"] != "")]
    return list(zip(prev["text"], prev["translation"]))


def run(url, tone="neutral", model_size="small", limit=None, max_no_speech=0.6, force=False,
        context_lines=3, retranslate=False):
    raw_path = miner.run_miner(url, model_size=model_size, force=force)
    video_id = os.path.splitext(os.path.basename(raw_path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{video_id}_{tone}.csv")
    srt_path = os.path.join(OUT_DIR, f"{video_id}_{tone}.srt")

    raw = pd.read_csv(raw_path)
    raw = raw[raw["no_speech_prob"] <= max_no_speech]          # 말소리가 아닐 가능성이 높은 줄 제외
    raw = raw[raw["text"].astype(str).str.strip().str.len() >= 2]
    if limit:
        raw = raw.head(limit)
    df = raw[["start_sec", "time", "text"]].copy()
    df["conf"] = raw["word_prob_mean"] if "word_prob_mean" in raw.columns else ""   # 낮을수록 인식이 불확실 (검수 대상)
    df["translation"] = ""

    if os.path.exists(csv_path) and not force and not retranslate:   # 이전에 번역한 줄 재사용
        done = pd.read_csv(csv_path).dropna(subset=["translation"])
        cache = dict(zip(done["text"], done["translation"]))
        df["translation"] = df["text"].map(cache).fillna("")
        print(f"이전 번역 {int((df['translation'] != '').sum())}줄을 재사용합니다.")

    todo = df.index[df["translation"] == ""]
    print(f"🌐 번역 중... ({len(todo)}줄, 톤: {tone}, 앞 문맥: {context_lines}줄)")
    for i, idx in enumerate(todo, 1):
        try:
            df.at[idx, "translation"] = translator.translate_slang(
                df.at[idx, "text"], mode="retrieval", temperature=0, tone=tone,
                context=get_context(df, idx, context_lines))
        except Exception as e:                                  # 한 줄 실패가 전체를 멈추지 않게
            print(f"  ⚠️ {df.at[idx, 'time']} 번역 실패: {e}")
        if i % 10 == 0:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")   # 중간 저장
            print(f"  {i}/{len(todo)}")
        time.sleep(0.3)  # 속도 제한 방지

    df["check"] = [("검수" if needs_check(t, c) else "") for t, c in zip(df["text"], df["conf"])]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_srt(df, srt_path)
    print(f"  인식이 불확실해 검수가 필요한 줄: {int((df['check'] == '검수').sum())}/{len(df)}")
    print(f"✅ 완료!\n  표: {csv_path}\n  자막: {srt_path}")
    return csv_path, srt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="유튜브 링크 -> 한국어 번역 자막")
    parser.add_argument("url", help="유튜브 영상 주소")
    parser.add_argument("--tone", choices=list(translator.TONES), default=translator.DEFAULT_TONE)
    parser.add_argument("--model", default="small", help="Whisper 모델: base < small < medium (기본 small)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N줄만 번역 (시험용)")
    parser.add_argument("--force", action="store_true", help="음성 인식과 번역을 처음부터 다시")
    parser.add_argument("--retranslate", action="store_true", help="음성 인식 결과는 그대로 두고 번역만 다시")
    parser.add_argument("--context", type=int, default=3, help="번역할 때 참고할 앞 문맥 줄 수 (기본 3, 0이면 사용 안 함)")
    args = parser.parse_args()
    run(args.url, tone=args.tone, model_size=args.model, limit=args.limit, force=args.force,
        context_lines=args.context, retranslate=args.retranslate)
