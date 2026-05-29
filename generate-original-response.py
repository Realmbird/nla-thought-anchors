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
    python generate-original-response.py
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

# Number of GSM8K test examples to process. None = all 1319.
N_EXAMPLES: int | None = None

# Batch size for forward + generate passes.
# 4× RTX 3090 (24 GB each): Qwen2.5-7B-Instruct in bfloat16 ≈ 14 GB on one GPU,
# leaving ~10 GB for KV cache + activations. Start at 8; raise to 16 if VRAM allows.
BATCH_SIZE = 8

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


# ── Batch helpers ─────────────────────────────────────────────────────────────

def build_prompts(tok: AutoTokenizer, questions: list[str]) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]


def process_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_examples: list[dict],
    batch_indices: list[int],
) -> list[dict]:
    questions = [ex["question"] for ex in batch_examples]
    golds = [ex["answer"] for ex in batch_examples]
    prompts = build_prompts(tok, questions)

    # Left-pad so all sequences end at the same position — last real token is always [:, -1, :].
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_lens = attention_mask.sum(dim=1).tolist()  # actual (non-pad) lengths

    # Pass 1: extract layer-20 residual stream at last prompt token.
    # hidden_states[0] = embedding output; hidden_states[k+1] = layer-k output.
    with torch.no_grad():
        fwd = model(input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
    # With left-padding the last real token is always at position -1.
    act_vecs: np.ndarray = (
        fwd.hidden_states[EXTRACT_LAYER + 1][:, -1, :]
        .float()
        .cpu()
        .numpy()
    )
    del fwd

    # Pass 2: generate responses.
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    rows = []
    for b, (question, gold, idx, act_vec, plen) in enumerate(
        zip(questions, golds, batch_indices, act_vecs, prompt_lens)
    ):
        total_len = input_ids.shape[1]
        # out_ids includes the (left-padded) prompt; skip it to get only new tokens.
        response = tok.decode(out_ids[b, total_len:], skip_special_tokens=True)
        rows.append({
            "question": question,
            "gold_answer": gold,
            "model_response": response,
            "is_correct": answers_match(response, gold),
            "activation_vector": act_vec,
            "example_idx": idx,
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Left-padding is required for batched generation with decoder-only models.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading GSM8K test split...")
    ds = load_dataset("gsm8k", "main", split="test")
    if N_EXAMPLES is not None:
        ds = ds.select(range(min(N_EXAMPLES, len(ds))))
    print(f"Processing {len(ds)} examples with batch_size={BATCH_SIZE}")

    correct_rows: list[dict] = []
    incorrect_rows: list[dict] = []
    total_processed = 0

    for batch_start in range(0, len(ds), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(ds))
        batch_examples = [ds[i] for i in range(batch_start, batch_end)]
        batch_indices = list(range(batch_start, batch_end))

        rows = process_batch(model, tok, batch_examples, batch_indices)
        for row in rows:
            (correct_rows if row["is_correct"] else incorrect_rows).append(row)
        total_processed += len(rows)

        done = len(correct_rows) + len(incorrect_rows)
        acc = len(correct_rows) / done * 100
        print(f"[{total_processed}/{len(ds)}]  acc={acc:.1f}%  "
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
            "question":          pa.array([r["question"] for r in rows]),
            "gold_answer":       pa.array([r["gold_answer"] for r in rows]),
            "model_response":    pa.array([r["model_response"] for r in rows]),
            "is_correct":        pa.array([r["is_correct"] for r in rows]),
            "activation_vector": pa.array(acts.tolist(), type=pa.list_(pa.float32())),
            "example_idx":       pa.array([r["example_idx"] for r in rows],
                                          type=pa.int32()),
        })

        out_path = OUTPUT_DIR / f"{split}.parquet"
        pq.write_table(table, str(out_path))
        print(f"Saved {len(rows)} rows → {out_path}")


# %%
if __name__ == "__main__":
    main()

# %%
