from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def _safe_read_dataframe(uploaded_file, fallback_path: Path) -> pd.DataFrame:
    """Read CSV/XLSX from upload; fallback to sample CSV if unavailable."""
    if uploaded_file is None:
        return pd.read_csv(fallback_path)

    file_name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))

    return pd.read_csv(fallback_path)


def load_bank_statement(uploaded_file=None) -> pd.DataFrame:
    """Load and normalize bank statement data for inflow/outflow analysis."""
    df = _safe_read_dataframe(uploaded_file, SAMPLE_DIR / "bank_statement.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "credit": "inflow",
        "debit": "outflow",
        "narration": "description",
    }
    df = df.rename(columns=rename_map)

    for col in ["inflow", "outflow", "balance"]:
        if col not in df.columns:
            df[col] = 0.0

    if "description" not in df.columns:
        df["description"] = "transaction"

    if "date" not in df.columns:
        df["date"] = pd.date_range("2025-04-01", periods=len(df), freq="7D")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["inflow"] = pd.to_numeric(df["inflow"], errors="coerce").fillna(0)
    df["outflow"] = pd.to_numeric(df["outflow"], errors="coerce").fillna(0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(method="ffill").fillna(0)

    return df


def load_gst_summary(uploaded_file=None) -> pd.DataFrame:
    """Load GST turnover data used for reconciliation checks."""
    df = _safe_read_dataframe(uploaded_file, SAMPLE_DIR / "gst_summary.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    if "month" not in df.columns:
        df["month"] = pd.date_range("2025-04-01", periods=len(df), freq="MS").strftime("%b-%Y")
    if "taxable_turnover" not in df.columns:
        df["taxable_turnover"] = 0
    if "gst_paid" not in df.columns:
        df["gst_paid"] = 0

    df["taxable_turnover"] = pd.to_numeric(df["taxable_turnover"], errors="coerce").fillna(0)
    df["gst_paid"] = pd.to_numeric(df["gst_paid"], errors="coerce").fillna(0)
    return df


def load_manual_notes(notes_text: Optional[str]) -> str:
    """Return manual officer notes, defaulting to sample notes when empty."""
    if notes_text and notes_text.strip():
        return notes_text.strip()

    sample_notes_path = SAMPLE_DIR / "qualitative_notes.txt"
    return sample_notes_path.read_text(encoding="utf-8")


def load_annual_report_text(uploaded_pdf=None) -> str:
    """Extract text from annual report PDF; fallback to bundled sample text."""
    if uploaded_pdf is None:
        return (SAMPLE_DIR / "annual_report_excerpt.txt").read_text(encoding="utf-8")

    raw_data = uploaded_pdf.getvalue()

    # First attempt with pdfplumber for better layout fidelity.
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw_data)) as pdf:
            for page in pdf.pages[:15]:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        joined = "\n".join(text_parts).strip()
        if joined:
            return joined
    except Exception:
        pass

    # Fallback to PyPDF2 for robustness across PDFs.
    try:
        reader = PdfReader(io.BytesIO(raw_data))
        text_parts = [(page.extract_text() or "") for page in reader.pages[:15]]
        joined = "\n".join(text_parts).strip()
        if joined:
            return joined
    except Exception:
        pass

    return (SAMPLE_DIR / "annual_report_excerpt.txt").read_text(encoding="utf-8")
