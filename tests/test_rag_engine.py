"""Unit tests for the local retrieval engine."""

from __future__ import annotations

import pytest

from modules.rag_engine import (
    THEME_QUERIES,
    build_local_rag_index,
    derive_rag_adjustments,
    retrieve_evidence_by_theme,
)

LITIGATION_TEXT = (
    "The company is subject to ongoing litigation including a commercial dispute "
    "before the court and an arbitration proceeding initiated by a supplier. "
    "A legal notice was served during the year and a penalty was levied by the "
    "regulator. Management believes the matters will be resolved without material "
    "impact, though the arbitration timeline remains uncertain and further legal "
    "notices may follow in the next financial year as disputes progress."
)


def _docs(text: str, source: str = "Annual Report"):
    return [
        {
            "source": source,
            "source_type": "PDF",
            "source_category": "Unstructured",
            "text": text,
        }
    ]


def test_index_builds_and_reports_ready():
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    assert index["status"] == "ready"
    assert index["chunk_count"] >= 1
    assert index["document_count"] == 1


def test_empty_documents_yield_empty_index():
    index = build_local_rag_index([{"source": "X", "text": ""}])
    assert index["status"] == "empty"
    assert index["chunk_count"] == 0


def test_short_text_is_discarded_as_noise():
    """Chunks under 25 words are dropped, so a one-liner indexes to nothing."""
    index = build_local_rag_index(_docs("Revenue grew."))
    assert index["chunk_count"] == 0


def test_chunks_carry_source_attribution():
    index = build_local_rag_index(_docs(LITIGATION_TEXT, source="Board Minutes"))
    for chunk in index["chunks"]:
        assert chunk["source"] == "Board Minutes"
        assert chunk["chunk_id"].startswith("CH-")


def test_litigation_theme_retrieves_litigation_text():
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    results = retrieve_evidence_by_theme(index, top_k=3)
    litigation = results["Litigation & Legal Disputes"]
    assert litigation["status"] == "evidence_found"
    assert litigation["snippets"][0]["score"] > 0


def test_all_themes_are_reported_even_when_empty():
    """Every theme must appear in the result so the UI can render a full table."""
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    results = retrieve_evidence_by_theme(index, top_k=3)
    assert set(results) == set(THEME_QUERIES)


def test_top_k_is_respected():
    long_text = " ".join([LITIGATION_TEXT] * 6)
    index = build_local_rag_index(_docs(long_text))
    results = retrieve_evidence_by_theme(index, top_k=2)
    for theme in results.values():
        assert len(theme["snippets"]) <= 2


def test_adjustments_are_capped_at_three():
    """A single document must not be able to trigger unlimited penalties."""
    noisy = " ".join(
        " ".join(terms) * 6 for terms in THEME_QUERIES.values()
    )
    index = build_local_rag_index(_docs(noisy))
    adjustments = derive_rag_adjustments(retrieve_evidence_by_theme(index))
    assert len(adjustments) <= 3


def test_adjustments_carry_evidence_attribution():
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    for adj in derive_rag_adjustments(retrieve_evidence_by_theme(index)):
        assert adj["evidence_source"]
        assert adj["evidence_chunk"].startswith("CH-")
        assert adj["evidence_snippet"]
        assert adj["points"] < 0


def test_no_evidence_yields_no_adjustments():
    index = build_local_rag_index(_docs("empty"))
    assert derive_rag_adjustments(retrieve_evidence_by_theme(index)) == []


# ---------------------------------------------------------------------------
# Phase 4.3 — IDF weighting and query tokenisation
# ---------------------------------------------------------------------------

BOILERPLATE = (
    "The company continued its operations during the financial year and the board "
    "of directors reviewed the performance of the company for the year under review. "
    "The company remains committed to the business and the financial year performance "
    "of the company was reviewed by the board during the financial year. "
)


def test_index_exposes_idf_weights():
    index = build_local_rag_index(_docs(BOILERPLATE * 4 + LITIGATION_TEXT))
    assert index["idf"], "No IDF table was built"
    assert all(weight > 0 for weight in index["idf"].values())


def test_boilerplate_is_weighted_below_distinctive_terms():
    """The point of IDF: words in every chunk must not outrank rare risk words."""
    index = build_local_rag_index(_docs(BOILERPLATE * 4 + LITIGATION_TEXT))
    idf = index["idf"]
    assert index["chunk_count"] > 1, "Need several chunks for IDF to mean anything"
    common = max(idf[t] for t in ("company", "year", "board") if t in idf)
    distinctive = min(idf[t] for t in ("arbitration", "penalty") if t in idf)
    assert distinctive > common, f"distinctive={distinctive} not above common={common}"


def test_chunks_carry_weighted_vectors():
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    for chunk in index["chunks"]:
        assert chunk["tfidf"], "Chunk has no tf-idf vector"
        assert set(chunk["tfidf"]) == set(chunk["vector"])


def test_multi_word_query_terms_can_match():
    """Regression: multi-word terms were stored as single dictionary keys.

    "related party" could never match a chunk, because chunks are tokenised into
    single words — so the most important governance signal in Indian corporate
    credit contributed nothing to retrieval.
    """
    text = (
        "The disclosures note that related party transactions were entered into "
        "during the year with entities controlled by the promoter group, and the "
        "audit committee reviewed each related party arrangement in detail before "
        "granting its approval for the transactions described in this section."
    )
    index = build_local_rag_index(_docs(text))
    results = retrieve_evidence_by_theme(index, top_k=3)
    governance = results["Governance & Promoter Concerns"]
    assert governance["status"] == "evidence_found"
    assert governance["snippets"][0]["score"] > 0.1


def test_empty_index_does_not_break_retrieval():
    index = build_local_rag_index([{"source": "X", "text": ""}])
    results = retrieve_evidence_by_theme(index)
    assert all(r["status"] == "no_evidence" for r in results.values())


# ---------------------------------------------------------------------------
# Regression guard for Phase 1.4: a stray `points = -1` used to overwrite the
# per-theme weight looked up one line earlier, flattening every theme to -1.
# ---------------------------------------------------------------------------


def test_theme_weights_are_differentiated():
    index = build_local_rag_index(_docs(LITIGATION_TEXT))
    adjustments = derive_rag_adjustments(retrieve_evidence_by_theme(index))
    litigation = [a for a in adjustments if a["signal"] == "Litigation & Legal Disputes"]
    assert litigation, "Expected a litigation adjustment from litigation-heavy text"
    assert litigation[0]["points"] == -2, (
        f"Litigation weighted at {litigation[0]['points']}, expected -2 per the mapping"
    )
