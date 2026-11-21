"""Build corpus DB, chunks, BM25, dense (BGE), and instruct (Qwen3) indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.config import (  # noqa: E402
    BM25_DIR,
    CHUNK_META_JSON,
    CHUNKS_JSONL,
    CORPUS_JSONL,
    DENSE_IDS,
    DENSE_INDEX,
    DENSE_MODEL_NAME,
    DOCS_DB,
    INDEX_DIR,
    INSTRUCT_IDS,
    INSTRUCT_INDEX,
    INSTRUCT_MODEL_NAME,
    ensure_data_dirs,
)
from src.corpus.loader import build_docs_db, chunk_text, iter_corpus_jsonl  # noqa: E402


def build_chunks(max_docs: int | None = None) -> int:
    ensure_data_dirs()
    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    n_chunks = 0
    with CHUNKS_JSONL.open("w", encoding="utf-8") as out:
        for i, doc in enumerate(iter_corpus_jsonl(CORPUS_JSONL)):
            if max_docs is not None and i >= max_docs:
                break
            pieces = chunk_text(doc.text)
            for j, piece in enumerate(pieces):
                chunk_id = f"{doc.doc_id}::{j}"
                row = {
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "chunk_index": j,
                    "text": piece,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                meta.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc.doc_id,
                        "chunk_index": j,
                    }
                )
                n_chunks += 1
            if (i + 1) % 5000 == 0:
                print(f"  chunked {i + 1} docs -> {n_chunks} chunks")
    CHUNK_META_JSON.write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {CHUNKS_JSONL} ({n_chunks} chunks)")
    return n_chunks


def _iter_chunks(path: Path = CHUNKS_JSONL):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_bm25() -> None:
    import bm25s
    import numpy as np

    ensure_data_dirs()
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    for row in _iter_chunks():
        texts.append(row["text"])
        chunk_ids.append(row["chunk_id"])
        doc_ids.append(row["doc_id"])
    print(f"Tokenizing {len(texts)} chunks for BM25...")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(BM25_DIR))
    mapping = {"chunk_ids": chunk_ids, "doc_ids": doc_ids}
    (BM25_DIR / "id_map.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )
    # keep a tiny numpy sidecart for length checks
    np.save(BM25_DIR / "n_chunks.npy", np.array([len(texts)], dtype=np.int64))
    print(f"Saved BM25 index to {BM25_DIR}")


def _encode_and_faiss(
    model_name: str,
    index_path: Path,
    ids_path: Path,
    batch_size: int = 64,
    normalize: bool = True,
    query_instruction: str | None = None,
    max_seq_length: int = 512,
) -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    texts: list[str] = []
    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    for row in _iter_chunks():
        texts.append(row["text"])
        chunk_ids.append(row["chunk_id"])
        doc_ids.append(row["doc_id"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder {model_name} on {device}...")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    dim = model.get_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    print(
        f"Encoding {len(texts)} chunks "
        f"(batch_size={batch_size}, max_seq_length={max_seq_length})"
    )

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encode {model_name}"):
        batch = texts[start : start + batch_size]
        # Passages are encoded without the query instruction
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=device,
        ).astype("float32")
        index.add(emb)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    from src.retrieval.faiss_io import write_faiss_index

    write_faiss_index(index, index_path)
    ids_path.write_text(
        json.dumps(
            {
                "chunk_ids": chunk_ids,
                "doc_ids": doc_ids,
                "model": model_name,
                "query_instruction": query_instruction,
            }
        ),
        encoding="utf-8",
    )
    print(f"Saved FAISS index ({index.ntotal} vectors) -> {index_path}")


def build_dense(batch_size: int = 64) -> None:
    _encode_and_faiss(
        DENSE_MODEL_NAME,
        DENSE_INDEX,
        DENSE_IDS,
        batch_size=batch_size,
    )


def build_instruct(batch_size: int = 32) -> None:
    from src.config import INSTRUCT_QUERY_PREFIX

    # max_seq_length=512 keeps VRAM bounded so larger batches are safe on 12GB.
    _encode_and_faiss(
        INSTRUCT_MODEL_NAME,
        INSTRUCT_INDEX,
        INSTRUCT_IDS,
        batch_size=batch_size,
        query_instruction=INSTRUCT_QUERY_PREFIX,
        max_seq_length=512,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes")
    parser.add_argument(
        "--stage",
        choices=["all", "corpus", "chunks", "bm25", "dense", "instruct"],
        default="all",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for faster smoke builds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encode batch size (default: 256 on CUDA, 64 on CPU)",
    )
    args = parser.parse_args()
    ensure_data_dirs()

    if args.batch_size is None:
        try:
            import torch

            args.batch_size = 256 if torch.cuda.is_available() else 64
        except Exception:
            args.batch_size = 64

    stages = (
        ["corpus", "chunks", "bm25", "dense", "instruct"]
        if args.stage == "all"
        else [args.stage]
    )

    if "corpus" in stages:
        if not CORPUS_JSONL.exists():
            raise SystemExit(
                f"Missing {CORPUS_JSONL}. Place the corpus JSONL at that path first."
            )
        n = build_docs_db(CORPUS_JSONL, DOCS_DB)
        print(f"Docs DB: {n} rows -> {DOCS_DB}")

    if "chunks" in stages:
        build_chunks(max_docs=args.max_docs)

    if "bm25" in stages:
        if not CHUNKS_JSONL.exists():
            build_chunks(max_docs=args.max_docs)
        build_bm25()

    if "dense" in stages:
        if not CHUNKS_JSONL.exists():
            build_chunks(max_docs=args.max_docs)
        build_dense(batch_size=args.batch_size)

    if "instruct" in stages:
        if not CHUNKS_JSONL.exists():
            build_chunks(max_docs=args.max_docs)
        # Slightly smaller than dense default; max_seq_length=512 limits VRAM.
        instruct_bs = min(args.batch_size, 64) if args.batch_size else 32
        build_instruct(batch_size=instruct_bs)

    print("Index build complete.")


if __name__ == "__main__":
    main()
