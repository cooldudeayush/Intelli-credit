from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _native_pdf_extract(raw_data: bytes, max_pages: int = 15) -> str:
    text_parts: List[str] = []

    try:
        with pdfplumber.open(io.BytesIO(raw_data)) as pdf:
            for page in pdf.pages[:max_pages]:
                text_parts.append(page.extract_text() or "")
        text = clean_text("\n".join(text_parts))
        if text:
            return text
    except Exception:
        pass

    try:
        reader = PdfReader(io.BytesIO(raw_data))
        text_parts = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
        text = clean_text("\n".join(text_parts))
        if text:
            return text
    except Exception:
        pass

    return ""


def _ocr_pdf_extract(raw_data: bytes, max_pages: int = 5) -> Tuple[str, str]:
    """OCR fallback using optional dependencies. Returns (text, message)."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except Exception:
        return "", "OCR dependencies unavailable (install pytesseract)."

    try:
        pdf = pdfium.PdfDocument(raw_data)
        pages = min(len(pdf), max_pages)
        ocr_parts: List[str] = []

        for idx in range(pages):
            page = pdf[idx]
            bitmap = page.render(scale=2).to_pil()
            ocr_parts.append(pytesseract.image_to_string(bitmap))

        text = clean_text("\n".join(ocr_parts))
        if not text:
            return "", "OCR ran but no text was detected."
        return text, "OCR fallback applied successfully."
    except Exception as exc:
        return "", f"OCR fallback failed: {exc}"


def extract_pdf_text_with_fallback(
    uploaded_file,
    fallback_text: str = "",
    min_chars_for_native: int = 120,
) -> Dict:
    """Unified PDF ingestion with native extraction and OCR fallback."""
    if uploaded_file is None:
        text = clean_text(fallback_text)
        return {
            "text": text,
            "uploaded": False,
            "parsed": bool(text),
            "method": "fallback" if text else "not_provided",
            "message": "Using fallback text" if text else "No source provided",
            "chars": len(text),
        }

    raw_data = uploaded_file.getvalue()
    native_text = _native_pdf_extract(raw_data)
    if len(native_text) >= min_chars_for_native:
        return {
            "text": native_text,
            "uploaded": True,
            "parsed": True,
            "method": "native",
            "message": "Native text extraction used",
            "chars": len(native_text),
        }

    ocr_text, ocr_message = _ocr_pdf_extract(raw_data)
    if ocr_text:
        return {
            "text": ocr_text,
            "uploaded": True,
            "parsed": True,
            "method": "ocr_fallback",
            "message": ocr_message,
            "chars": len(ocr_text),
        }

    fallback = clean_text(fallback_text)
    return {
        "text": fallback,
        "uploaded": True,
        "parsed": bool(fallback),
        "method": "fallback_after_failed_parse" if fallback else "failed",
        "message": ocr_message if ocr_message else "Parsing failed",
        "chars": len(fallback),
    }


def load_structured_or_pdf(uploaded_file, fallback_df: Optional[pd.DataFrame] = None) -> Dict:
    """Load CSV/XLSX as dataframe; if PDF, extract text via unified parser."""
    if uploaded_file is None:
        return {
            "kind": "fallback_dataframe" if fallback_df is not None else "missing",
            "dataframe": fallback_df.copy() if fallback_df is not None else pd.DataFrame(),
            "text": "",
            "parsed": fallback_df is not None,
            "method": "fallback" if fallback_df is not None else "not_provided",
        }

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
            return {"kind": "dataframe", "dataframe": df, "text": "", "parsed": True, "method": "csv"}
        except Exception:
            pass

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(data))
            return {"kind": "dataframe", "dataframe": df, "text": "", "parsed": True, "method": "excel"}
        except Exception:
            pass

    if name.endswith(".pdf"):
        parsed = extract_pdf_text_with_fallback(uploaded_file, "")
        return {
            "kind": "pdf_text",
            "dataframe": pd.DataFrame(),
            "text": parsed["text"],
            "parsed": parsed["parsed"],
            "method": parsed["method"],
            "message": parsed["message"],
        }

    return {
        "kind": "unsupported",
        "dataframe": fallback_df.copy() if fallback_df is not None else pd.DataFrame(),
        "text": "",
        "parsed": fallback_df is not None,
        "method": "unsupported_fallback" if fallback_df is not None else "unsupported",
    }


def _kw_count(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(t.count(k.lower()) for k in keywords)


def summarize_source_text(source: str, text: str) -> Dict:
    """Rule-based source summary for dashboard display."""
    t = text or ""
    return {
        "source": source,
        "litigation_mentions": _kw_count(t, ["litigation", "legal notice", "dispute", "court", "arbitration", "penalty"]),
        "guarantee_covenant_mentions": _kw_count(t, ["guarantee", "covenant", "security cover", "DSCR", "debt service"]),
        "governance_mentions": _kw_count(t, ["related party", "governance", "resignation", "audit", "fraud", "promoter"]),
        "debt_liability_mentions": _kw_count(t, ["debt", "liability", "borrowings", "interest", "default"]),
        "operational_stress_mentions": _kw_count(t, ["stress", "delay", "under capacity", "shutdown", "loss", "overdue"]),
    }


def summarize_structured_source(source: str, df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"source": source, "insight": "No structured rows available"}

    cols = [c.strip().lower() for c in df.columns]
    out = {"source": source, "rows": len(df), "columns": ", ".join(df.columns[:6])}

    if source == "ITR":
        revenue_like = [c for c in cols if any(x in c for x in ["income", "revenue", "turnover", "sales"])]
        tax_like = [c for c in cols if "tax" in c]
        out["revenue_columns"] = ", ".join(revenue_like[:3]) if revenue_like else "Not detected"
        out["tax_columns"] = ", ".join(tax_like[:3]) if tax_like else "Not detected"

    if source == "Shareholding Pattern":
        pct_col = next((c for c in df.columns if "share" in c.lower() and ("%" in c.lower() or "percent" in c.lower() or "holding" in c.lower())), None)
        if pct_col:
            series = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            if not series.empty:
                out["max_holder_pct"] = round(float(series.max()), 2)
                out["concentration_flag"] = "High" if float(series.max()) >= 50 else "Moderate"
        else:
            out["concentration_flag"] = "Unknown (percentage column not identified)"

    return out
