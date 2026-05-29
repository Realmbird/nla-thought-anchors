"""
Step 2: Verbalize activation vectors → NLA descriptions via SGLang actor.

Source: local parquet dir OR HuggingFace dataset repo.
Output: step2_nla/{split}.parquet — all original columns + nla_description.

Start SGLang first:
    uv run python -m sglang.launch_server \\
        --model-path kitft/nla-qwen2.5-7b-L20-av \\
        --port 30000 \\
        --disable-radix-cache \\
        --mem-fraction-static 0.85 \\
        --trust-remote-code

Then run:
    python generate-nla.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
from inference import EXPLANATION_RE, NLAClient


def resolve_local(repo_id_or_path: str) -> Path:
    """Return a local directory path, downloading from HF Hub if needed."""
    p = Path(repo_id_or_path)
    if p.exists():
        return p
    print(f"Downloading {repo_id_or_path} from HuggingFace Hub ...")
    return Path(snapshot_download(repo_id_or_path))

# ── Config ────────────────────────────────────────────────────────────────────

# "parquet" → read PARQUET_DIR/*.parquet (one file per split)
# "hf"      → stream from HF_REPO (each HF split becomes one output file)
SOURCE = "hf"
PARQUET_DIR = Path("step1_activations")
HF_REPO = "Realmbird/gsm8k-qwen2.5-7b-L20-activations"

ACTOR_CHECKPOINT = "kitft/nla-qwen2.5-7b-L20-av"   # HF repo; downloaded on first run
CRITIC_CHECKPOINT = "kitft/nla-qwen2.5-7b-L20-ar"  # optional — for round-trip MSE scoring
SGLANG_URL = "http://localhost:30000"
OUTPUT_DIR = Path("step2_nla")

# None = all examples; set to e.g. 50 for a quick test run
N_EXAMPLES: int | None = None

# Concurrent SGLang requests. Each holds an HTTP connection and ~(prompt_len * d_model * 4)
# bytes of embed memory. 8 is safe on 24 GB; raise to 16–32 for higher throughput.
CONCURRENCY = 8

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits_parquet(parquet_dir: Path) -> dict[str, pa.Table]:
    splits = {}
    for p in sorted(parquet_dir.glob("*.parquet")):
        splits[p.stem] = pq.read_table(str(p))
    if not splits:
        raise FileNotFoundError(f"no .parquet files in {parquet_dir}")
    return splits


def load_splits_hf(repo: str) -> dict[str, pa.Table]:
    from datasets import load_dataset
    ds = load_dataset(repo)
    # DatasetDict → {split_name: pa.Table}
    return {name: split.data.table for name, split in ds.items()}


def extract_vecs(table: pa.Table) -> np.ndarray:
    """activation_vector column (list<float32>) → float32 ndarray [N, d]."""
    flat = (table.column("activation_vector")
            .combine_chunks()   # ChunkedArray → single ListArray before flatten
            .flatten()
            .to_numpy(zero_copy_only=False)
            .astype(np.float32))
    return flat.reshape(len(table), -1)


# ── Async verbalization ───────────────────────────────────────────────────────

async def _verbalize_one(
    client: NLAClient,
    vec: np.ndarray,
    sem: asyncio.Semaphore,
    http: httpx.AsyncClient,
    sampling: dict,
) -> str:
    # Hold the semaphore through embed-build + HTTP to cap both concurrency and
    # peak embed-buffer memory. _build_embeds is pure CPU (no await) so it runs
    # atomically inside the event loop — no thread-safety issues.
    async with sem:
        embeds_np, _ = client._build_embeds(
            torch.as_tensor(vec, dtype=torch.float32), None
        )
        body = orjson.dumps(
            {"input_embeds": embeds_np, "sampling_params": sampling},
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        resp = await http.post(
            f"{client.sglang_url}/generate",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    resp.raise_for_status()
    out = resp.json()
    text = (out[0] if isinstance(out, list) else out)["text"]
    m = EXPLANATION_RE.search(text)
    if m is None:
        print(f"[warn] no <explanation> tags — Raw[:80]={text[:80]!r}")
        return text.strip()
    return m.group(1).strip()


async def verbalize_all(
    client: NLAClient,
    vecs: np.ndarray,
    concurrency: int,
    sampling: dict,
) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    done = [0]
    n = len(vecs)

    async def _tracked(i: int) -> str:
        result = await _verbalize_one(client, vecs[i], sem, http, sampling)
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == n:
            print(f"  [{done[0]}/{n}]")
        return result

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as http:
        # gather preserves order; tasks start immediately and yield at HTTP awaits
        return list(await asyncio.gather(*[_tracked(i) for i in range(n)]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    actor_dir = resolve_local(ACTOR_CHECKPOINT)
    print(f"Loading NLAClient from {actor_dir} ...")
    client = NLAClient(actor_dir, sglang_url=SGLANG_URL)

    print(f"Loading activation data (source={SOURCE!r}) ...")
    if SOURCE == "parquet":
        splits = load_splits_parquet(PARQUET_DIR)
    elif SOURCE == "hf":
        splits = load_splits_hf(HF_REPO)
    else:
        raise ValueError(f"SOURCE must be 'parquet' or 'hf', got {SOURCE!r}")

    sampling = {"temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS,
                "skip_special_tokens": False}

    for split_name, table in splits.items():
        if N_EXAMPLES is not None:
            table = table.slice(0, N_EXAMPLES)
        n = len(table)
        print(f"\n── {split_name}  ({n} examples) ──────────────────────────")

        vecs = extract_vecs(table)
        descriptions = asyncio.run(
            verbalize_all(client, vecs, CONCURRENCY, sampling)
        )

        out_table = table.append_column(
            "nla_description",
            pa.array(descriptions, type=pa.string()),
        )
        out_path = OUTPUT_DIR / f"{split_name}.parquet"
        pq.write_table(out_table, str(out_path))
        print(f"Saved {n} rows → {out_path}")


if __name__ == "__main__":
    main()
