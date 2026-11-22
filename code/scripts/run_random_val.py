"""Run SearchAgent on N randomly sampled validation questions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.agent.loop import SearchAgent  # noqa: E402
from src.config import VAL_JSONL, load_env  # noqa: E402
from src.llm.siliconflow import SiliconFlowClient  # noqa: E402
from src.retrieval.unified import UnifiedRetriever  # noqa: E402
from src.utils_em import normalize_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=CODE_ROOT / "val_prediction_random8.jsonl",
    )
    args = parser.parse_args()
    load_env()

    rows = [
        json.loads(line)
        for line in VAL_JSONL.open(encoding="utf-8")
        if line.strip()
    ]
    rng = random.Random(args.seed)
    n = min(args.n, len(rows))
    sample = rng.sample(rows, n)
    ids = [str(r.get("id") or r.get("query_id")) for r in sample]
    print(f"Sampled ids (seed={args.seed}): {ids}")

    llm = SiliconFlowClient()
    retriever = UnifiedRetriever(
        enable_bm25=True,
        enable_dense=True,
        enable_instruct=True,
        lazy=True,
    )
    agent = SearchAgent(llm=llm, retriever=retriever)

    correct = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(sample):
            qid = str(row.get("id") or row.get("query_id"))
            question = row.get("question") or row.get("query") or ""
            gold = row.get("answer")
            print(f"[{i + 1}/{n}] id={qid}")
            try:
                result = agent.run(question)
                answer = result.answer
                traj = result.trajectory
                tools = result.tool_calls
                tokens = result.total_tokens
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                answer = ""
                traj = [{"role": "system", "content": f"agent error: {exc}"}]
                tools = 0
                tokens = 0
            rec = {
                "id": qid,
                "question": question,
                "answer": answer,
                "trajectory": traj,
                "tool_calls": int(tools),
                "total_tokens": int(tokens),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if gold is not None:
                em = normalize_answer(answer) == normalize_answer(gold)
                correct += int(em)
                if em:
                    print("  EM=1")
                else:
                    print(f"  EM=0 pred={answer!r} gold={gold!r}")
            preview = (answer or "")[:80]
            print(f"  tools={tools} tokens={tokens} answer={preview!r}")

    elapsed = time.time() - t0
    print(f"Wrote {args.out} ({n} records) in {elapsed:.1f}s")
    print(f"Normalized EM: {correct}/{n} = {correct / n:.3f}")


if __name__ == "__main__":
    main()
