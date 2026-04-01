import os
import re
import pandas as pd
from faster_whisper import WhisperModel
import yt_dlp

# --- 환경 설정 ---
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --- 사용자 설정 ---
YOUTUBE_URL = "https://youtu.be/-7CYPrxatSA?si=5-6xslriWA8xw5oF" 

# 데이터 저장 경로 설정 (data 폴더 안의 파일)
DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "kai_slang_raw.csv")

MODEL_SIZE = "base"
DEVICE = "cpu" 

def download_audio(url):
    print("🚀 유튜브 오디오 다운로드 중...")
    ydl_opts = {
        'format': 'm4a/bestaudio/best', 
        'outtmpl': 'temp_audio.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "temp_audio.m4a"

def run_miner():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    audio_file = download_audio(YOUTUBE_URL)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8") 
    
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    extracted_data = []
    
    print("📝 모든 문장을 추출 중입니다...")
    for segment in segments:
        text = segment.text.strip()
        
        # 키워드 필터링 삭제! 모든 문장을 다 담습니다.
        timestamp = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
        
        extracted_data.append({
            "time": timestamp,
            "original_text": text,
            "translated_text": "" 
        })
            
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        file_exists = os.path.isfile(OUTPUT_PATH)
        # 덮어쓰지 않고 새로운 영상마다 다른 파일명으로 저장하는 게 관리하기 편할 수 있습니다.
        # video_id를 파일명에 넣는 것을 추천합니다.
        df.to_csv(OUTPUT_PATH, mode='a', index=False, encoding='utf-8-sig', header=not file_exists)
        print(f"✅ 완료! 총 {len(extracted_data)}개의 문장을 저장했습니다.")
        
    if os.path.exists(audio_file):
        os.remove(audio_file)

if __name__ == "__main__":
    run_miner()