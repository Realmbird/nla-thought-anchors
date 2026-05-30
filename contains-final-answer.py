#confirming ryan greenblatt's results at 7B after step2
# Ryan's numbers (Gemma, longer study):
#   correct   examples: NLA contains answer  80% of the time
#   incorrect examples: NLA contains (wrong) predicted answer  46% of the time
#                       NLA contains correct answer somewhat less often
# We replicate with Qwen2.5-7B: just check str(answer) in nla_description.
# %%
from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

# Match EXTRACT_POSITION in generate-nla.py — "hash" or "answer"
EXTRACT_POSITION = "hash"
NLA_DIR = Path(f"step2_nla_{EXTRACT_POSITION}")   # output of generate-nla.py

_ANS_RE = re.compile(r"####\s*([\-\d,\.]+)")
# %%

def parse_answer(text: str) -> str | None:
    m = _ANS_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None


def contains(nla: str, answer: str | None) -> bool:
    return answer is not None and answer in nla


def analyse(rows: list[dict]) -> dict:
    n = len(rows)
    contains_pred = sum(contains(r["nla"], r["pred"]) for r in rows)
    contains_gold = sum(contains(r["nla"], r["gold"]) for r in rows)
    return {
        "n": n,
        "contains_pred": contains_pred,
        "contains_pred_pct": contains_pred / n * 100 if n else 0,
        "contains_gold": contains_gold,
        "contains_gold_pct": contains_gold / n * 100 if n else 0,
    }


def load_rows(parquet_path: Path) -> list[dict]:
    table = pq.read_table(str(parquet_path))
    rows = []
    for i in range(len(table)):
        gold_raw  = table["gold_answer"][i].as_py()
        pred_raw  = table["model_response"][i].as_py()
        nla       = table["nla_description"][i].as_py() or ""
        correct   = table["is_correct"][i].as_py()
        rows.append({
            "gold": parse_answer(gold_raw),
            "pred": parse_answer(pred_raw),
            "nla":  nla,
            "correct": correct,
        })
    return rows

# %%
def main() -> None:
    parquet_files = sorted(NLA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {NLA_DIR} — run generate-nla.py first.")

    all_rows: list[dict] = []
    for p in parquet_files:
        all_rows.extend(load_rows(p))

    correct_rows   = [r for r in all_rows if r["correct"]]
    incorrect_rows = [r for r in all_rows if not r["correct"]]

    c = analyse(correct_rows)
    w = analyse(incorrect_rows)

    print(f"{'─'*55}")
    print(f"  Total examples : {len(all_rows)}")
    print(f"  Correct        : {c['n']}   Incorrect: {w['n']}")
    print(f"{'─'*55}")
    print(f"  CORRECT examples  (pred == gold)")
    print(f"    NLA contains answer : {c['contains_pred']}/{c['n']}  "
          f"= {c['contains_pred_pct']:.1f}%")
    print(f"    (Ryan @ Gemma: ~80%)")
    print()
    print(f"  INCORRECT examples  (pred != gold)")
    print(f"    NLA contains predicted (wrong) answer : "
          f"{w['contains_pred']}/{w['n']} = {w['contains_pred_pct']:.1f}%")
    print(f"    NLA contains correct  (gold)  answer : "
          f"{w['contains_gold']}/{w['n']} = {w['contains_gold_pct']:.1f}%")
    print(f"    (Ryan @ Gemma: ~46% contains wrong pred)")
    print(f"{'─'*55}")

    # ── Per-split breakdown if multiple files ─────────────────────────────────
    if len(parquet_files) > 1:
        print()
        for p in parquet_files:
            rows = load_rows(p)
            s = analyse(rows)
            print(f"  {p.stem:20s}  n={s['n']}  "
                  f"contains_pred={s['contains_pred_pct']:.1f}%  "
                  f"contains_gold={s['contains_gold_pct']:.1f}%")


# %%

if __name__ == "__main__":
    main()


