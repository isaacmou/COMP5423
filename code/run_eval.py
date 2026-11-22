"""Batch-run the agent on validation/test splits and write prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.agent.loop import SearchAgent  # noqa: E402
from src.config import (  # noqa: E402
    TEST_JSONL,
    VAL_JSONL,
    load_env,
)
from src.llm.siliconflow import SiliconFlowClient  # noqa: E402
from src.retrieval.unified import UnifiedRetriever  # noqa: E402
from src.utils_em import normalize_answer  # noqa: E402


def load_split(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_record(
    qid: str,
    question: str,
    answer: str,
    trajectory: list[dict[str, Any]],
    tool_calls: int,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "id": str(qid),
        "question": question,
        "answer": answer,
        "trajectory": trajectory,
        "tool_calls": int(tool_calls),
        "total_tokens": int(total_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMP5423 search agent evaluation")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: ../test_prediction.jsonl for test)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Disable dense/instruct backends (faster smoke test)",
    )
    parser.add_argument(
        "--compute-em",
        action="store_true",
        help="Print normalized EM on validation (requires answers)",
    )
    args = parser.parse_args()
    load_env()

    split_path = VAL_JSONL if args.split == "val" else TEST_JSONL
    if not split_path.exists():
        raise SystemExit(
            f"Missing {split_path}. Expected local file under data/."
        )

    out = args.out
    if out is None:
        out = (
            CODE_ROOT.parent / "test_prediction.jsonl"
            if args.split == "test"
            else CODE_ROOT / "val_prediction.jsonl"
        )

    rows = load_split(split_path)
    rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    llm = SiliconFlowClient(api_key=args.api_key) if args.api_key else SiliconFlowClient()
    retriever = UnifiedRetriever(
        enable_bm25=True,
        enable_dense=not args.bm25_only,
        enable_instruct=not args.bm25_only,
        lazy=True,
    )
    agent = SearchAgent(llm=llm, retriever=retriever)

    out.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total_with_gold = 0
    t0 = time.time()

    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            qid = str(row.get("id") or row.get("query_id"))
            question = row.get("question") or row.get("query") or ""
            gold: Optional[str] = row.get("answer")
            def _safe_print(msg: str) -> None:
                # Windows consoles (e.g. cp950) can choke on non-ASCII answers.
                try:
                    print(msg)
                except UnicodeEncodeError:
                    print(msg.encode("ascii", errors="replace").decode("ascii"))

            _safe_print(f"[{i + 1}/{len(rows)}] id={qid}")
            try:
                result = agent.run(question)
                rec = prediction_record(
                    qid=qid,
                    question=question,
                    answer=result.answer,
                    trajectory=result.trajectory,
                    tool_calls=result.tool_calls,
                    total_tokens=result.total_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                _safe_print(f"  ERROR: {exc}")
                rec = prediction_record(
                    qid=qid,
                    question=question,
                    answer="",
                    trajectory=[
                        {
                            "role": "system",
                            "content": f"agent error: {exc}",
                        }
                    ],
                    tool_calls=0,
                    total_tokens=0,
                )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if args.compute_em and gold is not None:
                total_with_gold += 1
                if normalize_answer(rec["answer"]) == normalize_answer(gold):
                    correct += 1
                    _safe_print("  EM=1")
                else:
                    _safe_print(f"  EM=0 pred={rec['answer']!r} gold={gold!r}")
            _safe_print(
                f"  tools={rec['tool_calls']} tokens={rec['total_tokens']} "
                f"answer={rec['answer'][:80]!r}"
            )

    elapsed = time.time() - t0
    print(f"Wrote {out} ({len(rows)} records) in {elapsed:.1f}s")
    if args.compute_em and total_with_gold:
        print(f"Normalized EM: {correct}/{total_with_gold} = {correct / total_with_gold:.3f}")


if __name__ == "__main__":
    main()
