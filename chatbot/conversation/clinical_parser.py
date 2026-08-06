"""Simple clinical history interpreter using rules/regex.

This module extracts a limited set of structured variables from free text clinical
histories and returns them with confidence scores so they can be applied to
the `PatientState`.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple, Any


def _find_age(text: str) -> Tuple[Any, float]:
    # Try explicit 'edad' or 'años' patterns
    m = re.search(r"edad[:\s]+(\d{1,3})", text, flags=re.I)
    if not m:
        m = re.search(r"(\d{1,3})\s*años\b", text, flags=re.I)
    if m:
        try:
            return int(m.group(1)), 0.9
        except Exception:
            return None, 0.0

    # Try DOB patterns dd/mm/yyyy or yyyy-mm-dd
    m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%d/%m/%Y")
            age = datetime.now().year - dt.year
            return age, 0.8
        except Exception:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d")
            age = datetime.now().year - dt.year
            return age, 0.8
        except Exception:
            pass

    return None, 0.0


def _find_gender(text: str) -> Tuple[Any, float]:
    if re.search(r"\b(masculino|hombre|varón|male)\b", text, flags=re.I):
        return 1, 0.9
    if re.search(r"\b(femenino|mujer|hembra|female)\b", text, flags=re.I):
        return 0, 0.9
    return None, 0.0


def _find_flag(text: str, keywords: list[str]) -> Tuple[int | None, float]:
    # If keyword appears with nearby negation words, return 0
    neg_pattern = re.compile(r"\b(no|sin|niega|niego|without)\b.{0,40}?\b({kw})\b", flags=re.I)
    for kw in keywords:
        # check explicit negation nearby
        neg = re.search(r"\b(no|sin|sin antecedentes de|sin historia de)\b.{0,40}?\b" + re.escape(kw) + r"\b", text, flags=re.I)
        if neg:
            return 0, 0.9
        if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.I):
            return 1, 0.9
    return None, 0.0


def _find_history_no_show(text: str) -> Tuple[Any, float]:
    # look for phrases like 'faltó 2 veces' or 'no-show 3'
    m = re.search(r"(falt[oó]|no-?show)\s*(?:[:,-])?\s*(\d{1,3})", text, flags=re.I)
    if m:
        try:
            return float(m.group(2)), 0.9
        except Exception:
            pass
    return None, 0.0


def _find_consultation_reason(text: str) -> Tuple[Any, float]:
    m = re.search(r"motivo[:\s]+(.{5,200})", text, flags=re.I)
    if m:
        return m.group(1).strip(), 0.8
    # fallback: try first sentence as reason
    s = text.strip().split(".\n")[0]
    if len(s) > 20:
        return s.strip(), 0.5
    return None, 0.0


def interpret_clinical_history(text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Return (extracted_values, confidences).

    extracted_values is a dict mapping PatientState field names to values.
    confidences maps field name to float in [0,1].
    """
    text = text or ""
    extracted: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}

    age, c = _find_age(text)
    if age is not None:
        extracted["age"] = age
        confidences["age"] = c

    gender, c = _find_gender(text)
    if gender is not None:
        extracted["gender_m"] = gender
        confidences["gender_m"] = c

    # flags
    for key, kws in (
        ("hypertension", ["hipertensi", "hipertensión", "hipertenso", "blood pressure"]),
        ("diabetes", ["diabetes", "diabético"]),
        ("alcoholism", ["alcohol", "alcoh" ]),
        ("handicap", ["discapaci", "handicap", "parálisis"]),
    ):
        val, conf = _find_flag(text, kws)
        if val is not None:
            extracted[key] = val
            confidences[key] = conf

    no_show, c = _find_history_no_show(text)
    if no_show is not None:
        extracted["history_no_show"] = no_show
        confidences["history_no_show"] = c

    reason, c = _find_consultation_reason(text)
    if reason:
        extracted["consultation_reason"] = reason
        confidences["consultation_reason"] = c

    return extracted, confidences
