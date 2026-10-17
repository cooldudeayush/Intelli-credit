from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List


THEME_QUERIES = {
    "Litigation & Legal Disputes": ["litigation", "legal notice", "dispute", "court", "arbitration", "penalty"],
    "Debt Obligations & Covenants": ["debt", "borrowings", "covenant", "guarantee", "repayment", "default", "interest"],
    "Governance & Promoter Concerns": ["governance", "promoter", "related party", "resignation", "audit", "fraud"],
    "Operational Stress Signals": ["stress", "delay", "under capacity", "shutdown", "loss", "overdue"],
    "Regulatory / Sector Risk Signals": ["regulatory", "compliance", "sector", "headwind", "approval", "policy"],
    "Contingent Liabilities & Negative Commentary": ["contingent", "liability", "qualified", "adverse", "negative", "uncertain"],
}


# Minimum tf-idf cosine similarity for retrieved evidence to affect the score.
# Chosen from the observed gap between incidental term overlap (a document that
# merely happens to contain one theme word scores ~0.02-0.05) and deliberate
# risk language (a document discussing the theme scores ~0.11-0.17). Sitting
# between the two keeps passing mentions out of the scorecard.
RELEVANCE_THRESHOLD = 0.10

# At most this many retrieved themes may adjust the score, so one noisy document
# cannot stack penalties across every dimension.
MAX_RAG_ADJUSTMENTS = 3


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", (text or "").lower())


def _chunk_text(text: str, chunk_size: int = 120, overlap: int = 25) -> List[str]:
    words = _tokenize(_normalize(text))
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 25:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks


def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0

    common = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _compute_idf(chunk_vectors: List[Counter]) -> Dict[str, float]:
    """Inverse document frequency over the chunk corpus.

    Without this, a term is worth what it is worth by frequency alone, so the
    boilerplate that saturates an annual report ("company", "year", "financial")
    outweighs the rare terms that actually carry risk signal. Smoothed so a term
    appearing in every chunk still contributes a little rather than exactly zero.
    """
    total = len(chunk_vectors)
    if total == 0:
        return {}

    document_frequency: Counter = Counter()
    for vector in chunk_vectors:
        document_frequency.update(vector.keys())

    return {
        term: math.log((1 + total) / (1 + df)) + 1.0
        for term, df in document_frequency.items()
    }


def _weight_by_idf(vector: Counter, idf: Dict[str, float], default_idf: float) -> Dict[str, float]:
    return {term: count * idf.get(term, default_idf) for term, count in vector.items()}


def build_local_rag_index(documents: List[Dict]) -> Dict:
    """Build a lightweight local vector index from unstructured document texts."""
    chunks = []
    chunk_id = 1

    for doc in documents:
        text = _normalize(doc.get("text", ""))
        if not text:
            continue

        for chunk_text in _chunk_text(text):
            chunks.append(
                {
                    "chunk_id": f"CH-{chunk_id:04d}",
                    "source": doc.get("source", "Unknown"),
                    "source_type": doc.get("source_type", "Unstructured"),
                    "source_category": doc.get("source_category", "Document"),
                    "text": chunk_text,
                    "vector": Counter(_tokenize(chunk_text)),
                }
            )
            chunk_id += 1

    # IDF needs the whole corpus, so weighting happens after every chunk exists.
    idf = _compute_idf([chunk["vector"] for chunk in chunks])
    default_idf = math.log(1 + len(chunks)) + 1.0 if chunks else 1.0
    for chunk in chunks:
        chunk["tfidf"] = _weight_by_idf(chunk["vector"], idf, default_idf)

    return {
        "chunks": chunks,
        "document_count": len([d for d in documents if (d.get("text") or "").strip()]),
        "chunk_count": len(chunks),
        "idf": idf,
        "default_idf": default_idf,
        "status": "ready" if chunks else "empty",
    }


def retrieve_evidence_by_theme(index: Dict, top_k: int = 3) -> Dict:
    """Retrieve top evidence chunks for each credit-risk theme."""
    chunks = index.get("chunks", [])
    idf = index.get("idf", {})
    default_idf = index.get("default_idf", 1.0)
    results = {}

    for theme, query_terms in THEME_QUERIES.items():
        # Tokenise the query the same way chunks are tokenised. Multi-word terms
        # like "related party" were previously kept as single dictionary keys,
        # so they could never match a chunk and contributed nothing.
        query_vec = Counter(_tokenize(" ".join(query_terms)))
        query_weighted = _weight_by_idf(query_vec, idf, default_idf)
        scored = []

        for chunk in chunks:
            score = _cosine_similarity(query_weighted, chunk.get("tfidf", {}))
            if score > 0:
                scored.append(
                    {
                        "score": round(score, 4),
                        "chunk_id": chunk["chunk_id"],
                        "source": chunk["source"],
                        "source_type": chunk["source_type"],
                        "source_category": chunk["source_category"],
                        "snippet": chunk["text"][:280],
                    }
                )

        scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        status = "evidence_found" if scored else "no_evidence"

        summary = "No significant evidence retrieved."
        if scored:
            summary = f"Retrieved {len(scored)} supporting snippets from {len(set(s['source'] for s in scored))} document(s)."

        results[theme] = {
            "status": status,
            "summary": summary,
            "snippets": scored,
            "contributes_to_risk": True if scored else False,
        }

    return results


def derive_rag_adjustments(theme_results: Dict) -> List[Dict]:
    """Map retrieved evidence themes to deterministic Five C impacts."""
    # Weights reflect how directly each theme evidences credit risk. Litigation
    # and governance findings bear on Character most sharply — they are
    # statements about the borrower's conduct, not its circumstances — so they
    # carry double the weight of the contextual themes.
    mapping = {
        "Litigation & Legal Disputes": ("Character", -2),
        "Debt Obligations & Covenants": ("Capacity", -1),
        "Governance & Promoter Concerns": ("Character", -2),
        "Operational Stress Signals": ("Capacity", -1),
        "Regulatory / Sector Risk Signals": ("Conditions", -1),
        "Contingent Liabilities & Negative Commentary": ("Capital", -1),
    }

    adjustments: List[Dict] = []
    for theme, result in theme_results.items():
        if not result.get("snippets"):
            continue

        top = result["snippets"][0]
        if float(top.get("score", 0)) < RELEVANCE_THRESHOLD:
            continue

        factor, points = mapping.get(theme, ("Conditions", -1))
        adjustments.append(
            {
                "source": "Unstructured RAG",
                "source_type": "unstructured_rag",
                "signal": theme,
                "factor": factor,
                "points": points,
                "reason": f"Theme evidence found in {top['source']} ({top['chunk_id']}).",
                "evidence_source": top["source"],
                "evidence_chunk": top["chunk_id"],
                "evidence_snippet": top["snippet"],
            }
        )
        if len(adjustments) >= MAX_RAG_ADJUSTMENTS:
            break

    return adjustments
