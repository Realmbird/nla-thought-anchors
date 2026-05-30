"""
Sanity check: are the NLA descriptions from kitft/nla-qwen2.5-7b-L20-av
actually informative, or just generic math-prompt boilerplate?

Loads Realmbird/nla-av-explanations from HF, produces:
  - printed examples (correct + incorrect)
  - phrase frequency bar chart
  - description length distribution
  - TF-IDF cosine similarity heatmap (100-sample subset)
  - correct vs incorrect description overlap comparison
"""
# %%
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO = "Realmbird/nla-av-explanations"
OUT_DIR = Path("sanity_check_plots")
OUT_DIR.mkdir(exist_ok=True)

_ANS_RE = re.compile(r"####\s*([\-\d,\.]+)")

def parse_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = _ANS_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None

# %%  ── Load data ──────────────────────────────────────────────────────────────

print(f"Loading {REPO} ...")
ds = load_dataset(REPO)
# flatten all splits into one list
rows = []
for split in ds.values():
    for ex in split:
        rows.append(ex)

print(f"Total rows: {len(rows)}")

correct   = [r for r in rows if r["is_correct"]]
incorrect = [r for r in rows if not r["is_correct"]]
nlas      = [r["nla_description"] or "" for r in rows]

# %%  ── Examples ────────────────────────────────────────────────────────────────

def show_examples(subset: list[dict], label: str, n: int = 5) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label} (showing {n})")
    print(f"{'─'*60}")
    for r in subset[:n]:
        gold = parse_answer(r["gold_answer"])
        pred = parse_answer(r["model_response"])
        nla  = (r["nla_description"] or "").strip()
        print(f"  gold={gold}  pred={pred}")
        print(f"  Q:   {r['question'][:120]}")
        print(f"  NLA: {nla[:300]}")
        print()

show_examples(correct,   "CORRECT examples")
show_examples(incorrect, "INCORRECT examples")

# %%  ── Phrase frequency ────────────────────────────────────────────────────────

# Split each description into overlapping 5-grams of words
def ngrams(text: str, n: int) -> list[str]:
    words = text.lower().split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

all_5grams: list[str] = []
for nla in nlas:
    all_5grams.extend(ngrams(nla, 5))

top_phrases = Counter(all_5grams).most_common(20)
phrases, counts = zip(*top_phrases)
pct = [c / len(nlas) * 100 for c in counts]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(phrases)), pct, color="#4C72B0")
ax.set_yticks(range(len(phrases)))
ax.set_yticklabels(phrases, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("% of descriptions containing phrase")
ax.set_title("Top 20 five-word phrases in NLA descriptions\n"
             "(high frequency = descriptions are generic/repetitive)")
for bar, p in zip(bars, pct):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{p:.0f}%", va="center", fontsize=8)
plt.tight_layout()
out = OUT_DIR / "phrase_frequency.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
plt.close()

# %%  ── Description length distribution ──────────────────────────────────────

lengths_correct   = [len(r["nla_description"] or "") for r in correct]
lengths_incorrect = [len(r["nla_description"] or "") for r in incorrect]

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, max(max(lengths_correct), max(lengths_incorrect)) + 50, 40)
ax.hist(lengths_correct,   bins=bins, alpha=0.6, label=f"correct (n={len(correct)})",
        color="#2196F3")
ax.hist(lengths_incorrect, bins=bins, alpha=0.6, label=f"incorrect (n={len(incorrect)})",
        color="#F44336")
ax.set_xlabel("NLA description length (chars)")
ax.set_ylabel("Count")
ax.set_title("Description length: correct vs incorrect\n"
             "(similar distributions → descriptions don't distinguish outcome)")
ax.legend()
ax.axvline(np.median(lengths_correct),   color="#2196F3", linestyle="--", linewidth=1.5)
ax.axvline(np.median(lengths_incorrect), color="#F44336", linestyle="--", linewidth=1.5)
plt.tight_layout()
out = OUT_DIR / "length_distribution.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close()

# %%  ── TF-IDF cosine similarity heatmap (100-sample subset) ──────────────────

rng = np.random.default_rng(42)
idx = rng.choice(len(nlas), size=min(100, len(nlas)), replace=False)
sample_nlas = [nlas[i] for i in idx]
sample_labels = ["C" if rows[i]["is_correct"] else "W" for i in idx]

tfidf = TfidfVectorizer(max_features=500, stop_words="english")
vecs  = tfidf.fit_transform(sample_nlas).toarray()
sim   = cosine_similarity(vecs)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim, cmap="hot", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="cosine similarity")
ax.set_title("Pairwise TF-IDF cosine similarity (100-sample)\n"
             "If descriptions are generic, the matrix should be uniformly bright")
ax.set_xlabel("example index")
ax.set_ylabel("example index")
ax.tick_params(labelbottom=False, labelleft=False)
plt.tight_layout()
out = OUT_DIR / "tfidf_similarity.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close()

# %%  ── Summary stats ──────────────────────────────────────────────────────────

boilerplate = sum(1 for nla in nlas if "structured math problem" in nla.lower())
mean_sim = (sim.sum() - len(sim)) / (len(sim)**2 - len(sim))  # off-diagonal mean

print(f"\n{'═'*60}")
print(f"  SANITY CHECK SUMMARY")
print(f"{'═'*60}")
print(f"  Total descriptions        : {len(nlas)}")
print(f"  Contain 'structured math' : {boilerplate}/{len(nlas)} = "
      f"{boilerplate/len(nlas)*100:.1f}%")
print(f"  Mean off-diag TF-IDF sim  : {mean_sim:.3f}  "
      f"(1.0 = identical, 0.0 = unrelated)")
print(f"  Median length correct     : {np.median(lengths_correct):.0f} chars")
print(f"  Median length incorrect   : {np.median(lengths_incorrect):.0f} chars")
print(f"\n  Verdict: if 'structured math' % is high (>60%) and mean sim")
print(f"  is high (>0.5), the actor is outputting boilerplate — it's not")
print(f"  extracting answer-relevant information from the activation.")
print(f"{'═'*60}")
print(f"\nAll plots saved to {OUT_DIR}/")

# %%
