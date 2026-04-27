# 🎤 Project Kai-Talk: Hyper-Localized Slang Translator
**Bridging the Gap: AAVE/NY Slang to Korean Gen-Z & Internet Slang**

유튜브 자동 번역은 Kai Cenat이 **"You're cooked"**라고 할 때 **"당신은 요리되었습니다"**라고 번역합니다.

Project Kai-Talk은 이러한 문화적·언어적 한계를 극복하기 위해 탄생했습니다. 미국 스트리밍 씬의 날것 그대로인 슬랭을 한국 인터넷 방송(치지직, 아프리카TV) 및 커뮤니티(디시, 펨코) 감성으로 **초현지화(Hyper-localization)** 하는 데이터셋과 번역 엔진을 제공합니다.

---

## 🚀 Why Kai-Talk? (Benchmark)

기존 번역기와의 압도적인 차이를 확인하세요.

| Original Text (Kai Cenat) | YouTube Auto-Translate | Kai-Talk (Ours) |
|---|---|---|
| "Yo, he's actually cooked." | "요, 그는 실제로 요리되었습니다." | "와 저 새끼 진짜 나락 갔네(좆됐네)." |
| "Stop glazing that man!" | "그 남자를 유리하게 하지 마세요!" | "야, 걔 후빨 뇌절 좀 그만해 진짜." |
| "I gotta stand on business." | "나는 사업을 시작해야 합니다." | "나도 내 가오는 살려야지 ㅇㅋ?" |
| "He's tweaking right now." | "그는 지금 미세 조정 중입니다." | "쟤 지금 맛탱이 갔네 ㅋㅋㅋ" |

---

## 🛠️ System Architecture

이 프로젝트는 데이터 수집부터 최종 애플리케이션까지의 전체 파이프라인을 포함합니다.

**Data Mining:**
Faster-Whisper (STT)를 활용해 수 시간 분량의 라이브 스트림에서 슬랭 키워드가 포함된 문장을 자동 추출합니다.

**Dataset Curation:**
추출된 Raw Data를 Claude 3.5와 GPT-4o를 통해 레이블링하고, 한국 인방 고인물의 검수를 거쳐 찰진 병렬 코퍼스(Parallel Corpus)를 구축했습니다.

**Slang Translator:**
구축된 데이터셋을 기반으로 Few-shot Learning 기법을 적용, 새로운 문장이 입력되어도 데이터셋의 '말투'와 '맥락'을 그대로 복제하여 번역합니다.

---

## 📦 Project Structure

```
Kai-Talk/
├── src/
│   ├── miner.py        # 유튜브 오디오 추출 및 슬랭 필터링 엔진
│   └── translator.py   # 데이터셋 기반 슬랭 변환기 (LLM)
├── data/
│   ├── kai_slang_raw.csv         # 추출된 원본 스크립트
│   └── kai_slang_translated.csv  # [핵심 가치] 초현지화 번역 데이터셋
├── requirements.txt    # 의존성 라이브러리
└── .env                # API Key 관리 (Private)
```

---

## 🔧 Installation & Usage

### 1. Requirements

```bash
pip install -r requirements.txt
```

> **Note:** FFmpeg 설치가 필요합니다.

### 2. Run Slang Miner

원하는 유튜브 URL에서 슬랭 데이터를 채굴합니다.

```bash
python src/miner.py
```

### 3. Run Kai-Talk Translator

우리의 데이터셋으로 학습된 찰진 번역기를 실행합니다. (OpenAI 또는 Groq API 필요)

```bash
python src/translator.py
```

---

## 🧠 Core Technologies

- **Speech-to-Text:** OpenAI Whisper (Faster-Whisper implementation)
- **LLM Engine:** GPT-4o-mini, Llama-3.3-70b (via Groq)
- **Data Engineering:** Pandas, Regular Expressions
- **Labeling Strategy:** Few-shot Contextual Prompting

---

## 📈 Future Roadmap

- [ ] 데이터셋 규모 확장 (1,000+ Sentence Pairs)
- [ ] 실시간 트위치/유튜브 자막 오버레이 개발
- [ ] 스트리머별 고유 말투(Style Transfer) 커스텀 모델 파인튜닝

---

## 🤝 Contribution

미국 슬랭과 한국 유행어의 가교 역할을 함께하실 분들을 환영합니다. 새로운 슬랭 데이터셋 추가는 언제든 **Pull Request**를 보내주세요!
