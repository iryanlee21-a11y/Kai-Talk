"""Kai-Talk Miner: 유튜브 영상 -> 음성 인식(STT) -> 문장별 CSV

사용법:
    python src/miner.py "https://youtu.be/VIDEO_ID"
    python src/miner.py "https://youtu.be/VIDEO_ID" --model medium --force

영상마다 data/raw/<video_id>.csv 파일을 하나씩 만듭니다.
이미 처리한 영상은 건너뛰고(--force로 다시 처리), 어느 영상에서 나온 문장인지 video_id로 추적할 수 있습니다.
"""
import argparse
import csv
import os
import re
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows/Anaconda의 OpenMP 중복 로드 오류 방지

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIELDS = ["video_id", "start_sec", "time", "text", "avg_logprob", "no_speech_prob", "word_prob_mean", "word_prob_min"]

# 음성 인식 힌트: 자주 나오는 이름과 슬랭을 알려주면 'Yeezy'가 'easy'로 나오는 오류가 줄어듭니다.
# (너무 길게 쓰면 힌트 단어가 결과에 섞여 나올 수 있어요. 필요에 따라 수정하세요.)
INITIAL_PROMPT = (
    "Kai Cenat stream. Yeezy, Kanye, Vultures, Drake, Kendrick, Adidas, Nike, NOCTA, "
    "glazing, cooked, rizz, gyatt, no cap, deadass, on god, motion, lock in."
)


def extract_video_id(url):
    """유튜브 주소에서 11자리 영상 ID를 꺼냅니다. (?si=... 같은 추적 값은 무시)"""
    match = re.search(r"(?:youtu\.be/|[?&]v=|/shorts/|/live/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"유튜브 영상 ID를 찾을 수 없습니다: {url}")
    return match.group(1)


def format_time(seconds):
    """초를 'mm:ss' 문자열로 바꿉니다. (예: 754 -> '12:34')"""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def segments_to_rows(video_id, segments):
    """Whisper 결과를 CSV 한 줄씩으로 바꿉니다. 인식 신뢰도도 함께 저장합니다.

    avg_logprob / no_speech_prob 는 약 30초 구간 단위 값이라 같은 구간의 줄은 값이 모두 같습니다.
    줄마다 다른 신뢰도는 word_prob_mean / word_prob_min (단어별 확률의 평균 / 최솟값)을 보세요.
    """
    rows = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        probs = [w.probability for w in (seg.words or [])]
        rows.append({
            "video_id": video_id,
            "start_sec": round(seg.start, 1),
            "time": format_time(seg.start),
            "text": text,
            "avg_logprob": round(seg.avg_logprob, 3),   # 0에 가까울수록 인식이 확실함
            "no_speech_prob": round(seg.no_speech_prob, 3),  # 높으면 말소리가 아닐 가능성
            "word_prob_mean": round(sum(probs) / len(probs), 3) if probs else "",  # 낮을수록 이 줄의 인식이 불확실함
            "word_prob_min": round(min(probs), 3) if probs else "",   # 가장 불확실한 단어
        })
    return rows


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def download_audio(url, workdir):
    """오디오만 내려받고, 실제로 저장된 파일 경로를 돌려줍니다."""
    import yt_dlp

    options = {
        "format": "m4a/bestaudio/best",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def run_miner(url, model_size="small", force=False):
    video_id = extract_video_id(url)
    out_path = os.path.join(RAW_DIR, f"{video_id}.csv")
    if os.path.exists(out_path) and not force:
        print(f"이미 처리한 영상입니다: {out_path} (다시 하려면 --force)")
        return out_path

    from faster_whisper import WhisperModel

    with tempfile.TemporaryDirectory() as tmp:  # 임시 오디오는 끝나면 자동 삭제
        print("🚀 오디오 다운로드 중...")
        audio_file = download_audio(url, tmp)

        print(f"📝 음성 인식 중... (모델: {model_size}, 영상 길이에 따라 시간이 걸립니다)")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            audio_file,
            language="en",        # 언어 자동 감지 오류 방지
            beam_size=5,
            vad_filter=True,      # 말소리가 없는 구간에서 지어내는 문장을 줄임
            word_timestamps=True, # 줄마다 단어별 확률을 얻기 위해 필요
            initial_prompt=INITIAL_PROMPT,
        )
        rows = segments_to_rows(video_id, segments)  # 실제 인식은 여기서 진행됨

    save_rows(out_path, rows)
    print(f"✅ 완료! {len(rows)}개 문장을 저장했습니다: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="유튜브 영상에서 영어 발화를 문장별로 추출합니다.")
    parser.add_argument("url", help="유튜브 영상 주소")
    parser.add_argument("--model", default="small", help="Whisper 모델 크기: base < small < medium (기본 small)")
    parser.add_argument("--force", action="store_true", help="이미 처리한 영상도 다시 처리")
    args = parser.parse_args()
    run_miner(args.url, model_size=args.model, force=args.force)
