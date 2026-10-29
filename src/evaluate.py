"""Kai-Talk 평가: 예시 선택 방식(없음 / 랜덤 / 검색)에 따라 번역 품질이 어떻게 달라지는지 비교합니다.

사용법 (프로젝트 루트에서):
    python src/evaluate.py run --tone neutral    # 150개 번역 생성 -> data/eval/ 에 저장
    (data/eval/score_sheet_neutral.csv 를 열어 meaning, naturalness 열을 채우기)
    python src/evaluate.py summarize --tone neutral   # 조건별 평균 계산

채점 기준
    meaning      : 0 = 의미가 틀림, 1 = 대체로 맞지만 어색하거나 일부 누락, 2 = 정확함
    naturalness  : 1 ~ 5 (1 = 번역투/어색함, 5 = 한국어로 매우 자연스러움)
채점표는 조건(none/random/retrieval)을 가린 채 섞여 있습니다. 편향 없이 채점하려면 조건을 보지 마세요.
"""
import argparse
import os
import random
import time

import pandas as pd

import translator

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
EVAL_SET = os.path.join(BASE, "eval_set.csv")
OUT_DIR = os.path.join(BASE, "eval")
CONDITIONS = ["none", "random", "retrieval"]


def run(tone, seed=0):
    os.makedirs(OUT_DIR, exist_ok=True)
    items = pd.read_csv(EVAL_SET)
    rows = []
    for _, item in items.iterrows():
        for cond in CONDITIONS:
            output = translator.translate_slang(item["text"], mode=cond, temperature=0, seed=seed, tone=tone)
            rows.append({"id": item["id"], "type": item["type"], "text": item["text"],
                         "condition": cond, "output": output})
            time.sleep(0.3)  # 속도 제한 방지
        print(f"{item['id']}/{len(items)} 완료")

    full = pd.DataFrame(rows)
    random.Random(seed).shuffle(rows)  # 조건을 가리기 위해 섞음
    for n, r in enumerate(rows, 1):
        r["row_id"] = n
    key = pd.DataFrame(rows)[["row_id", "id", "type", "condition"]]
    sheet = pd.DataFrame(rows)[["row_id", "text", "output"]].assign(meaning="", naturalness="")

    full.to_csv(os.path.join(OUT_DIR, f"outputs_{tone}.csv"), index=False, encoding="utf-8-sig")
    key.to_csv(os.path.join(OUT_DIR, f"key_{tone}.csv"), index=False, encoding="utf-8-sig")
    sheet.to_csv(os.path.join(OUT_DIR, f"score_sheet_{tone}.csv"), index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUT_DIR}\\score_sheet_{tone}.csv 를 열어 meaning, naturalness 열을 채우세요.")


def summarize(tone):
    key = pd.read_csv(os.path.join(OUT_DIR, f"key_{tone}.csv"))
    sheet = pd.read_csv(os.path.join(OUT_DIR, f"score_sheet_{tone}.csv"))
    df = key.merge(sheet[["row_id", "meaning", "naturalness"]], on="row_id")
    df[["meaning", "naturalness"]] = df[["meaning", "naturalness"]].apply(pd.to_numeric, errors="coerce")
    if df[["meaning", "naturalness"]].isna().any().any():
        print(f"⚠️ 아직 채점하지 않은 칸이 {int(df['meaning'].isna().sum())}개 있습니다. 채점된 것만으로 계산합니다.")
    df["correct"] = (df["meaning"] == 2).astype(float).where(df["meaning"].notna())

    def table(g):
        t = g.groupby("condition").agg(n=("meaning", "count"), 의미정확률=("correct", "mean"),
                                       의미평균=("meaning", "mean"), 자연스러움=("naturalness", "mean"))
        return t.reindex(CONDITIONS).round(2)

    print("\n=== 전체 ===");  print(table(df))
    for ty in ["slang", "plain"]:
        print(f"\n=== {ty} 문장 ===");  print(table(df[df['type'] == ty]))
    out = os.path.join(OUT_DIR, f"summary_{tone}.csv")
    table(df).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kai-Talk 번역 평가")
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--tone", choices=list(translator.TONES), default=translator.DEFAULT_TONE)
    args = parser.parse_args()
    run(args.tone) if args.command == "run" else summarize(args.tone)
