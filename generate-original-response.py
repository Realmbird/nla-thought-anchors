"""
Step 1: Run Qwen2.5-7B-Instruct on GSM8K, extract layer-20 activations,
sort by correct/incorrect, save to parquet.

Model : Qwen/Qwen2.5-7B-Instruct  (28 layers, d_model=3584, extract layer 20)
Data  : GSM8K test split (1319 examples)
Output: step1_activations/correct.parquet
        step1_activations/incorrect.parquet

Each parquet has columns:
  question          string
  gold_answer       string
  model_response    string
  is_correct        bool
  activation_vector list<float32>  len=3584  — residual stream after layer 20,
                                               last token of input prompt
  example_idx       int32

Run:
    uv add datasets          # if not already installed
    uv run python generate-original-response.py
"""
# %%
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
# %%
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXTRACT_LAYER = 20          # residual stream after transformer layer 20 (0-indexed)
D_MODEL = 3584
MAX_NEW_TOKENS = 512
OUTPUT_DIR = Path("step1_activations")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "At the end, write your final answer as: #### <number>"
)

# ── GSM8K answer parsing ──────────────────────────────────────────────────────
_ANS_RE = re.compile(r"####\s*([\-\d,\.]+)")


def _parse_answer(text: str) -> str | None:
    m = _ANS_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None


def answers_match(response: str, gold: str) -> bool:
    pred = _parse_answer(response)
    ref = _parse_answer(gold)
    if pred is None or ref is None:
        return False
    try:
        return float(pred) == float(ref)
    except ValueError:
        return pred == ref
# %%

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading GSM8K test split...")
    ds = load_dataset("gsm8k", "main", split="test")

    correct_rows: list[dict] = []
    incorrect_rows: list[dict] = []

    for i, ex in enumerate(ds):
        question: str = ex["question"]
        gold: str = ex["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = input_ids.shape[1]

        # Pass 1: extract layer-20 residual stream at last prompt token.
        # hidden_states[0] = embedding output; hidden_states[k+1] = layer-k output.
        with torch.no_grad():
            fwd = model(input_ids, output_hidden_states=True, use_cache=False)
        act_vec: np.ndarray = (
            fwd.hidden_states[EXTRACT_LAYER + 1][0, -1, :]
            .float()
            .cpu()
            .numpy()
        )
        del fwd

        # Pass 2: generate the model's response.
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out_ids[0, prompt_len:], skip_special_tokens=True)

        correct = answers_match(response, gold)
        row = {
            "question": question,
            "gold_answer": gold,
            "model_response": response,
            "is_correct": correct,
            "activation_vector": act_vec,
            "example_idx": i,
        }
        (correct_rows if correct else incorrect_rows).append(row)

        if (i + 1) % 25 == 0:
            done = len(correct_rows) + len(incorrect_rows)
            acc = len(correct_rows) / done * 100
            print(f"[{i + 1}/{len(ds)}]  acc={acc:.1f}%  "
                  f"correct={len(correct_rows)}  incorrect={len(incorrect_rows)}")

    total = len(correct_rows) + len(incorrect_rows)
    print(f"\nFinal: {len(correct_rows)}/{total} correct "
          f"({len(correct_rows) / total * 100:.1f}%)")

    # ── Save parquet ──────────────────────────────────────────────────────────
    for split, rows in [("correct", correct_rows), ("incorrect", incorrect_rows)]:
        if not rows:
            print(f"No {split} examples — skipping.")
            continue

        acts = np.stack([r["activation_vector"] for r in rows]).astype(np.float32)

        table = pa.table({
            "question":         pa.array([r["question"] for r in rows]),
            "gold_answer":      pa.array([r["gold_answer"] for r in rows]),
            "model_response":   pa.array([r["model_response"] for r in rows]),
            "is_correct":       pa.array([r["is_correct"] for r in rows]),
            "activation_vector": pa.array(acts.tolist(), type=pa.list_(pa.float32())),
            "example_idx":      pa.array([r["example_idx"] for r in rows],
                                         type=pa.int32()),
        })

        out_path = OUTPUT_DIR / f"{split}.parquet"
        pq.write_table(table, str(out_path))
        print(f"Saved {len(rows)} rows → {out_path}")

# %%
if __name__ == "__main__":
    main()

# %%

# %%
