"""인식 신뢰도(conf) 기준이 틀린 줄을 얼마나 잘 잡아내는지 확인합니다.

사용법:
    1) 결과 CSV를 복사해서 이름을 바꿉니다. (pipeline을 다시 돌리면 원본이 덮어써지므로 꼭 복사하세요)
         Copy-Item data/output/<영상ID>_neutral.csv data/output/<영상ID>_labeled.csv
    2) 복사본을 엑셀이나 VS Code로 열고, 영상을 들으면서 'wrong' 열을 추가해
       음성 인식(text)이 틀린 줄에 Y 를 적습니다. (맞는 줄은 비워 둠)
    3) python src/eval_conf.py data/output/<영상ID>_labeled.csv

출력: 기준값(conf 이하이고 단어 수가 min_words 이상인 줄을 '검수 필요'로 표시)별로
      - 표시된 줄 수, 그중 실제 틀린 줄 수(정밀도)
      - 전체 틀린 줄 중 잡아낸 비율(재현율)
"""
import argparse

import pandas as pd


def evaluate(df, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8), min_words_list=(1, 3)):
    wrong = df["wrong"].fillna("").astype(str).str.strip().str.upper().eq("Y")
    words = df["text"].astype(str).str.split().str.len()
    conf = pd.to_numeric(df["conf"], errors="coerce")
    total_wrong = int(wrong.sum())
    print(f"전체 {len(df)}줄 중 인식이 틀린 줄: {total_wrong}줄 ({total_wrong / len(df):.0%})\n")
    rows = []
    for mw in min_words_list:
        for th in thresholds:
            flagged = (conf < th) & (words >= mw)
            caught = int((flagged & wrong).sum())
            rows.append({
                "최소단어수": mw, "기준(conf<)": th, "표시된줄": int(flagged.sum()), "그중틀림": caught,
                "정밀도": round(caught / flagged.sum(), 2) if flagged.sum() else None,
                "재현율": round(caught / total_wrong, 2) if total_wrong else None,
            })
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conf 기준 평가")
    parser.add_argument("csv", help="'wrong' 열을 추가한 결과 CSV")
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    if "wrong" not in data.columns:
        raise SystemExit("'wrong' 열이 없습니다. 틀린 줄에 Y 를 적은 'wrong' 열을 추가해 주세요.")
    evaluate(data)
