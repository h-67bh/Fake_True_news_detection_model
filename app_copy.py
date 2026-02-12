# ==== Imports (consolidated) ====
import os
import re
import json
import time
import uuid
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st

from google.oauth2 import service_account
from google.cloud import translate
from langdetect import detect

# Transformers (used by model helpers defined elsewhere in this file)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification

# Stop-words (for Explain tab token filtering)
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP
except Exception:
    # Minimal fallback so the app doesn't crash if scikit-learn isn't present
    _SK_STOP = {
        "a","an","the","and","or","but","if","then","else","when","at","by","for","in","of","on","to","up","down","from","as",
        "is","are","was","were","be","been","being","with","this","that","these","those",
        "it","its","i","you","he","she","we","they","them","me","my","your","yours","his","her","their","theirs","our","ours"
    }
    
# ==== MODEL CHECKPOINT CONFIG (NEW) ====
# Discover locally saved fine-tuned checkpoints in the project root (portable).
# You can override with env vars: XLMR_CKPT, ROBERTA_EN_CKPT, BERT_EN_CKPT, MBERT_CKPT.
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

def _discover_local_ckpt(keys: list[str]) -> str | None:
    """
    Return the first subdirectory of ROOT whose name contains *all* key substrings
    (case-insensitive) and looks like a HF checkpoint (has config.json).
    """
    try:
        for p in ROOT.iterdir():
            if not p.is_dir():
                continue
            name = p.name.lower()
            if all(k in name for k in keys) and (p / "config.json").exists():
                return str(p.resolve())
    except Exception:
        pass
    return None

# Try common names like: bert_fakenews_model, roberta_fakenews_model, mbert_fakenews_model, xlm-roberta*_model
MODEL_PATHS = {
    "xlmr":   os.getenv("XLMR_CKPT")       or _discover_local_ckpt(["xlm", "roberta", "model"]),
    "roberta":os.getenv("ROBERTA_EN_CKPT") or _discover_local_ckpt(["roberta", "model"]),
    "bert":   os.getenv("BERT_EN_CKPT")    or _discover_local_ckpt(["bert", "model"]),
    "mbert":  os.getenv("MBERT_CKPT")      or _discover_local_ckpt(["mbert", "model"]),
}

# ==== Devices ====
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Lazy loaders for classifiers (cached) ====
 # Load mBERT classifier from local checkpoint or Hugging Face; cached for reuse
@st.cache_resource
def _load_mbert():
    """
    Load mBERT classifier.
    Priority:
    1) Local fine-tuned checkpoint at MODEL_PATHS['mbert']
    2) HF base 'bert-base-multilingual-uncased'
    """
    try:
        ckpt = MODEL_PATHS.get("mbert")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-multilingual-uncased", num_labels=2
            ).to(_DEVICE)
        mdl.eval()
        return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

 # Load English BERT classifier for the MT path; cached
@st.cache_resource
def _load_bert_en():
    """
    Load English BERT classifier for the MT path.
    Priority:
    1) Local fine-tuned checkpoint at MODEL_PATHS['bert']
    2) HF base 'bert-base-uncased'
    """
    try:
        ckpt = MODEL_PATHS.get("bert")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            ).to(_DEVICE)
        mdl.eval()
        return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

 # Load XLM-R classifier for multilingual direct scoring; cached
@st.cache_resource
def _load_xlmr_cls():
    """
    Load XLM-R classifier (Gate-1).
    Priority:
    1) Local fine-tuned checkpoint at MODEL_PATHS['xlmr']
    2) HF base 'xlm-roberta-base'
    """
    try:
        ckpt = MODEL_PATHS.get("xlmr")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "xlm-roberta-base", num_labels=2
            ).to(_DEVICE)
        mdl.eval()
        return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

 # Load English RoBERTa classifier used after translation; cached
@st.cache_resource
def _load_roberta_en():
    """
    Load English RoBERTa classifier for the MT path (Gate-2).
    Priority:
    1) Local fine-tuned checkpoint at MODEL_PATHS['roberta']
    2) HF base 'roberta-base'
    """
    try:
        ckpt = MODEL_PATHS.get("roberta")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("roberta-base")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            ).to(_DEVICE)
        mdl.eval()
        return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

# Instantiate globals used elsewhere (prevents NameError on import)
try:
    tokenizer_m, model_m, device_m = _load_mbert()
    tokenizer_e, model_e, device_e = _load_bert_en()
    tokenizer_x, model_x, device_x = _load_xlmr_cls()
    tokenizer_r, model_r, device_r = _load_roberta_en()
except Exception:
    tokenizer_m = model_m = device_m = None
    tokenizer_e = model_e = device_e = None
    tokenizer_x = model_x = device_x = None
    tokenizer_r = model_r = device_r = None

 # Convert model logits to probabilities with softmax
def _softmax(logits: torch.Tensor) -> np.ndarray:
    if isinstance(logits, np.ndarray):
        x = torch.tensor(logits)
    else:
        x = logits
    probs = torch.softmax(x, dim=-1).detach().cpu().numpy()
    return probs

 # Compute predictive entropy as an uncertainty measure
def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())

 # Compute margin between top two class probabilities (confidence proxy)
def _top2_margin(probs: np.ndarray) -> float:
    if probs.ndim == 2:
        p = probs[0]
    else:
        p = probs
    s = np.sort(p)[::-1]
    if len(s) < 2:
        return 0.0
    return float(s[0] - s[1])

 # Run tokenizer+model forward pass and return raw logits
def _logits(tokenizer, model, device, text: str) -> torch.Tensor:
    if tokenizer is None or model is None:
        raise RuntimeError("Requested model/tokenizer not available.")
    enc = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        return out.logits

 # Convert logits to predicted label, max probability, and full probability vector
def _pred_from_logits(logits: torch.Tensor):
    probs = _softmax(logits)
    if probs.ndim == 2:
        p = probs[0]
    else:
        p = probs
    y = int(np.argmax(p))
    p_max = float(np.max(p))
    return y, p_max, p

 # Score English text with RoBERTa and apply temperature calibration
def _predict_roberta_en_calibrated(en_text: str, pre: dict | None = None) -> dict:
    T = _get_temperature("roberta", (pre or {}).get("calibration_ids"))
    logits = _logits(tokenizer_r, model_r, device_r, en_text)
    logits_cal = _apply_temperature(logits, T)
    y, p_max, p = _pred_from_logits(logits_cal)
    return {
        "path": "MT→RoBERTa",
        "label": y,
        "prob": p_max,
        "probs": p,
        "logits": logits.detach().cpu().numpy().tolist() if hasattr(logits, "detach") else None,
        "logits_cal": logits_cal.detach().cpu().numpy().tolist() if hasattr(logits_cal, "detach") else None,
        "temperature": float(T),
        "calibration_id": (pre or {}).get("calibration_ids", {}).get("roberta"),
        "text_en": en_text,
    }

 # Score English text with BERT and apply temperature calibration
def _predict_bert_en_calibrated(en_text: str, pre: dict | None = None) -> dict:
    T = _get_temperature("bert", (pre or {}).get("calibration_ids"))
    logits = _logits(tokenizer_e, model_e, device_e, en_text)
    logits_cal = _apply_temperature(logits, T)
    y, p_max, p = _pred_from_logits(logits_cal)
    return {
        "path": "BERT(en)",
        "label": y,
        "prob": p_max,
        "probs": p,
        "logits": logits.detach().cpu().numpy().tolist() if hasattr(logits, "detach") else None,
        "logits_cal": logits_cal.detach().cpu().numpy().tolist() if hasattr(logits_cal, "detach") else None,
        "temperature": float(T),
        "calibration_id": (pre or {}).get("calibration_ids", {}).get("bert"),
        "text_en": en_text,
    }

 # Multilingual path: run XLM-R on original text and report confidence
def predict_xlmr(text: str) -> dict:
    pre = prechecks(text)
    T = _get_temperature("xlmr", pre.get("calibration_ids"))
    logits = _logits(tokenizer_x, model_x, device_x, text)
    logits_cal = _apply_temperature(logits, T)
    y, p_max, p = _pred_from_logits(logits_cal)

    # NEW: uncertainty metrics for the explicit XLM-R report
    H = _entropy(p)
    m = _top2_margin(p)

    return {
        "path": "XLM-RoBERTa",
        "label": y,
        "prob": p_max,
        "probs": p,
        "entropy": float(H),         
        "margin": float(m),         
        "logits": logits.detach().cpu().numpy().tolist() if hasattr(logits, "detach") else None,
        "logits_cal": logits_cal.detach().cpu().numpy().tolist() if hasattr(logits_cal, "detach") else None,
        "temperature": float(T),
        "calibration_id": pre.get("calibration_ids", {}).get("xlmr"),
    }

 # Length features: ratio and absolute log penalty between source and target
def _len_features(src: str, tgt: str) -> tuple[float, float]:
    ls = max(1, len(src))
    lt = max(1, len(tgt))
    ratio = lt / ls
    l = float(abs(np.log(ratio)))
    return ratio, l

 # Load LaBSE sentence embedding model if available; cached
@st.cache_resource
def _labse_model():
    try:
        from sentence_transformers import SentenceTransformer
        mdl = SentenceTransformer("sentence-transformers/LaBSE")
        return mdl
    except Exception:
        return None

 # Compute cosine similarity using LaBSE embeddings (None if unavailable)
def _labse_cosine(a: str, b: str) -> float | None:
    mdl = _labse_model()
    if mdl is None:
        return None
    try:
        v = mdl.encode([a, b], normalize_embeddings=True)
        return float(np.clip(np.dot(v[0], v[1]), -1.0, 1.0))
    except Exception:
        return None

 # Consistency check: translate EN back to source and compare semantics
def _round_trip_cosine_from_en(src_text: str, en_text: str, src_lang: str | None) -> float | None:
    if not src_lang:
        return None
    try:
        back = gcp_translate(en_text, target=src_lang)
        c = _labse_cosine(src_text, back)
        return c
    except Exception:
        return None

 # Load NLLB-200 translation model and tokenizer; cached
@st.cache_resource
def _load_nllb():
    try:
        from transformers import AutoTokenizer as _HF_Tok, AutoModelForSeq2SeqLM
        tok = _HF_Tok.from_pretrained("facebook/nllb-200-distilled-600M")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(_DEVICE)
        mdl.eval()
        return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

 # Tell whether NLLB model successfully loaded
def _nllb_available() -> bool:
    tok, mdl, dev = _load_nllb()
    return tok is not None and mdl is not None

# Minimal ISO→NLLB tag map (fallback to eng_Latn)
_NLLB_TAGS = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "hi": "hin_Deva", "bn": "ben_Beng", "vi": "vie_Latn", "id": "ind_Latn",
    "sw": "swh_Latn", "zh": "zho_Hans", "zh-cn": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
}

 # Map ISO language code to NLLB language tag with safe fallback
def _nllb_tag_from_iso(code: str | None) -> str:
    if not code:
        return "eng_Latn"
    c = code.lower()
    return _NLLB_TAGS.get(c, _NLLB_TAGS.get(c.split('-')[0], "eng_Latn"))

 # MT→RoBERTa path: translate, score with RoBERTa, weight by quality signals
def predict_roberta_plus_mt_best(text: str):
    pre = prechecks(text)
    src_lang = (pre.get("lang") or "").lower()
    qe_bins  = pre.get("mt_qe_bins", {}) or {}
    engines  = pre.get("engines", {}) or {}

    candidates = []
    def _score_candidate(engine_key: str, en_text: str, qe_bin: str | None):
        # (1) Calibrated RoBERTa on English
        pred = _predict_roberta_en_calibrated(en_text, pre)
        p_e  = float(pred["prob"])

        # (2) Quality weights
        w_qe = _qe_weight_from_bin(qe_bin or "OK")
        r_e  = _round_trip_cosine_from_en(text, en_text, src_lang)
        _, l_e = _len_features(text, en_text)
        bt_ok = (r_e is not None and r_e >= MT_BT_R0) and (l_e <= MT_LEN_L0)
        w_bt  = 1.0 if bt_ok else 0.6

        c_e = _labse_cosine(text, en_text)
        w_sem = float(np.clip(c_e, MT_SEM_CMIN, 1.0)) if c_e is not None else 1.0
        w_len = float(np.exp(-l_e))

        # (3) Scores
        score_basic = float(p_e * w_qe * w_bt)
        score_ext   = float(score_basic * w_sem * w_len)
        score_used  = score_ext  # extended (still in [0,1])

        return {
            "path": "MT→RoBERTa",
            "engine": engine_key,
            "engine_version": ((engines.get(engine_key) or {}).get("version")),
            "label": int(pred["label"]),

            # Effective confidence used by Decision Module (and for route comparisons)
            "prob": score_used,

            # Keep raw RoBERTa signal for observability
            "roberta_prob": p_e,
            "probs": pred.get("probs"),
            "logits": pred.get("logits"),
            "logits_cal": pred.get("logits_cal"),

            # Signals & weights
            "qe_bucket": (qe_bin or "OK"),
            "w_QE": float(w_qe),
            "round_trip_cosine": (None if r_e is None else float(r_e)),
            "len_penalty_abs_log": float(l_e),
            "bt_ok": bool(bt_ok),
            "w_BT": float(w_bt),
            "semantic_cosine": (None if c_e is None else float(c_e)),
            "w_sem": float(w_sem),
            "w_len": float(w_len),

            "score_basic": score_basic,
            "score_ext": score_ext,

            "text_en": en_text,
            "temperature": float(pred.get("temperature", 1.0)),
            "calibration_id": pred.get("calibration_id"),
        }

    # (C.1) MT adapters: Google & NLLB
    # Mono-English fast path (no MT); treat as synthetic engine 'mono'
    try:
        if src_lang == "en" or looks_english(text):
            candidates.append(_score_candidate("mono", text, qe_bin="Good"))
    except Exception:
        pass

    # Google MT
    try:
        if (engines.get("google") or {}).get("allowed", True):
            t_g = gcp_translate(
                text, target="en",
                source=(src_lang or None) if src_lang and src_lang != "en" else None
            )
            candidates.append(_score_candidate("google", t_g, qe_bins.get("google")))
    except Exception:
        pass

    # NLLB MT (only if available)
    try:
        nllb_meta = (engines.get("nllb") or {})
        if nllb_meta.get("allowed", True) and nllb_meta.get("available", False):
            t_n = nllb_translate(
                text, target="en",
                source=(src_lang or None) if src_lang and src_lang != "en" else None
            )
            candidates.append(_score_candidate("nllb", t_n, qe_bins.get("nllb")))
    except Exception:
        pass

    if not candidates:
        # Fallback: plain RoBERTa on the original text (assume EN)
        pred_fallback = _predict_roberta_en_calibrated(text, pre)
        return {
            "path": "MT→RoBERTa",
            "explicit_mode": "roBERTa (+MT)",
            "engine": "fallback",
            "engine_version": None,
            "label": int(pred_fallback["label"]),
            "prob": float(pred_fallback["prob"]),
            "probs": pred_fallback.get("probs"),
            "logits": pred_fallback.get("logits"),
            "logits_cal": pred_fallback.get("logits_cal"),
            "qe_bucket": "OK",
            "w_QE": 1.0,
            "bt_ok": True,
            "w_BT": 1.0,
            "semantic_cosine": None,
            "w_sem": 1.0,
            "w_len": 1.0,
            "round_trip_cosine": None,
            "len_penalty_abs_log": 0.0,
            "score_basic": float(pred_fallback["prob"]),
            "score_ext": float(pred_fallback["prob"]),
            "text_en": text,
            "temperature": float(pred_fallback.get("temperature", 1.0)),
            "calibration_id": pred_fallback.get("calibration_id"),
            "scores_all": [
                {
                    "engine": "fallback",
                    "qe_bucket": "OK",
                    "score": float(pred_fallback["prob"]),
                    "roberta_prob": float(pred_fallback["prob"]),
                    "w_QE": 1.0,
                    "w_BT": 1.0,
                    "bt_ok": True,
                    "w_sem": 1.0,
                    "w_len": 1.0,
                    "round_trip_cosine": None,
                    "len_penalty_abs_log": 0.0
                }
            ]
        }

    # Pick best engine by effective (weighted) confidence
    best = max(candidates, key=lambda r: r.get("prob", 0.0))
    best["explicit_mode"] = "roBERTa (+MT)"

    # Compact per-engine scoreboard (for logs / UI debug if needed)
    best["scores_all"] = [
        {
            "engine": c.get("engine"),
            "qe_bucket": c.get("qe_bucket"),
            "score": float(c.get("prob", 0.0)),                 # S_e used
            "roberta_prob": float(c.get("roberta_prob", 0.0)),  # raw \hat p_e
            "w_QE": float(c.get("w_QE", 1.0)),
            "w_BT": float(c.get("w_BT", 1.0)),
            "w_sem": float(c.get("w_sem", 1.0)),
            "w_len": float(c.get("w_len", 1.0)),
            "bt_ok": bool(c.get("bt_ok", False)),
            "round_trip_cosine": (None if c.get("round_trip_cosine") is None else float(c.get("round_trip_cosine"))),
            "len_penalty_abs_log": float(c.get("len_penalty_abs_log", 0.0)),
        }
        for c in candidates
    ]
    best["path"] = "MT→RoBERTa"
    return best

 # MT→BERT path: translate, score with BERT, weight by quality signals
def predict_bert_plus_mt_best(text: str):
    """
    BERT(+MT) path with per-engine scoring (mirrors Gate-2, but classifier is BERT).
    Returns a record that includes the chosen MT engine and effective (weighted) confidence.
    """
    pre = prechecks(text)
    src_lang = (pre.get("lang") or "").lower()
    qe_bins  = pre.get("mt_qe_bins", {}) or {}
    engines  = pre.get("engines", {}) or {}
    candidates = []

    def _score_candidate(engine_key: str, en_text: str, qe_bin: str | None):
        pred = _predict_bert_en_calibrated(en_text, pre)
        p_e  = float(pred["prob"])  # calibrated BERT prob

        w_qe = _qe_weight_from_bin(qe_bin or "OK")
        r_e  = _round_trip_cosine_from_en(text, en_text, src_lang)
        _, l_e = _len_features(text, en_text)
        bt_ok = (r_e is not None and r_e >= MT_BT_R0) and (l_e <= MT_LEN_L0)
        w_bt  = 1.0 if bt_ok else 0.6

        c_e = _labse_cosine(text, en_text)
        w_sem = float(np.clip(c_e, MT_SEM_CMIN, 1.0)) if c_e is not None else 1.0
        w_len = float(np.exp(-l_e))

        score_basic = float(p_e * w_qe * w_bt)
        score_ext   = float(score_basic * w_sem * w_len)
        score_used  = score_ext

        return {
            "path": "BERT+MT",
            "engine": engine_key,
            "engine_version": ((engines.get(engine_key) or {}).get("version")),
            "label": int(pred["label"]),
            "prob": score_used,                   # effective (weighted) confidence
            "bert_prob": p_e,                     # raw calibrated BERT prob (display)
            "probs": pred.get("probs"),
            "logits": pred.get("logits"),
            "logits_cal": pred.get("logits_cal"),
            "qe_bucket": (qe_bin or "OK"),
            "w_QE": float(w_qe),
            "round_trip_cosine": (None if r_e is None else float(r_e)),
            "len_penalty_abs_log": float(l_e),
            "bt_ok": bool(bt_ok),
            "w_BT": float(w_bt),
            "semantic_cosine": (None if c_e is None else float(c_e)),
            "w_sem": float(w_sem),
            "w_len": float(w_len),
            "score_basic": score_basic,
            "score_ext": score_ext,
            "text_en": en_text,
            "temperature": float(pred.get("temperature", 1.0)),
            "calibration_id": pred.get("calibration_id"),
        }

    # English fast path (no MT)
    try:
        if src_lang == "en" or looks_english(text):
            pred = _predict_bert_en_calibrated(text, pre)
            return {**pred, "path": "BERT(en)", "engine": None, "engine_version": None, "bt_ok": True}
    except Exception:
        pass

    # Google
    try:
        if (engines.get("google") or {}).get("allowed", True):
            t_g = gcp_translate(
                text, target="en",
                source=(src_lang or None) if src_lang and src_lang != "en" else None
            )
            candidates.append(_score_candidate("google", t_g, qe_bins.get("google")))
    except Exception:
        pass

    # NLLB (only if available)
    try:
        nllb_meta = (engines.get("nllb") or {})
        if nllb_meta.get("allowed", True) and nllb_meta.get("available", False):
            t_n = nllb_translate(
                text, target="en",
                source=(src_lang or None) if src_lang and src_lang != "en" else None
            )
            candidates.append(_score_candidate("nllb", t_n, qe_bins.get("nllb")))
    except Exception:
        pass

    if not candidates:
        pred = _predict_bert_en_calibrated(text, pre)
        return {**pred, "path": "BERT(en)", "engine": None, "engine_version": None, "bt_ok": True}

    return max(candidates, key=lambda r: r.get("prob", 0.0))

# ===== Adaptive routing thresholds (NEW) =====
HIGH_CONF = 0.75   # "high" confidence for conflict checks
LOW_CONF  = 0.55   # "usable" confidence floor
# --- Gate-2 scoring thresholds (back-translation & semantics) ---
MT_BT_R0    = 0.60  # round-trip cosine r_e threshold (higher = more consistent)
MT_LEN_L0   = 0.55  # |log length-ratio| threshold (lower = better)
MT_SEM_CMIN = 0.55  # min semantic cosine used for w_sem clip

 # Convert numeric label to human-readable class name
def _label_name(y: int) -> str:
    return "Fake" if int(y) == 1 else "True"

 # Check if two predictions strongly disagree with high confidence
def _conflict_high(a: dict, b: dict) -> bool:
    """Opposite labels and both above HIGH_CONF."""
    return (a.get("label") != b.get("label")) and (a.get("prob", 0.0) >= HIGH_CONF) and (b.get("prob", 0.0) >= HIGH_CONF)


 # Create a tiny punctuation tweak for stability testing
def _small_perturb(text: str) -> str:
    """
    Minimal, semantics-preserving tweak to test stability.
    Appends a period if not present, or swaps a punctuation mark if possible.
    """
    t = str(text)
    # If already ends with a period, swap the last period/question/exclamation with another
    if t.endswith("."):
        return t[:-1] + "!"
    elif t.endswith("!"):
        return t[:-1] + "."
    elif t.endswith("?"):
        return t[:-1] + "."
    # If ends with other punctuation, swap to period
    elif t and t[-1] in ",;:":
        return t[:-1] + "."
    # Otherwise, append a period
    elif t and not t[-1] in ".!?":
        return t + "."
    else:
        return t

# ===== Decision Module (Gate‑1 vs Gate‑2) =====
 # Flip test: perturb text for a voter and see if the label changes
def _run_flip_check(voter: str, base_record: dict, text: str, en_text: str | None, pre: dict) -> tuple[float, bool]:

    # Decide which text to perturb
    if voter == "X":
        perturbed = _small_perturb(text)
        try:
            pred = predict_xlmr(perturbed)
        except Exception:
            return 1.0, False
    elif voter == "M":
        if not en_text:
            return 1.0, False
        perturbed = _small_perturb(en_text)
        try:
            pred = _predict_roberta_en_calibrated(perturbed, pre)
        except Exception:
            return 1.0, False
    elif voter == "B":
        if not en_text:
            return 1.0, False
        perturbed = _small_perturb(en_text)
        try:
            pred = predict_bert_en(perturbed, do_translate=False)
        except Exception:
            return 1.0, False
    elif voter == "m":
        perturbed = _small_perturb(text)
        try:
            pred = predict_mbert(perturbed)
        except Exception:
            return 1.0, False
    else:
        return 1.0, False
    flipped = int(pred.get("label", -1)) != int(base_record.get("label", -2))
    return (0.7 if flipped else 1.0), flipped

 # Tie-breaker: combine XLM-R, RoBERTa, BERT, and mBERT with stability weights
def _tie_breaker_layer(
    text: str,
    r_x: dict,
    r_m: dict,
    pre: dict | None = None,
    dm_meta: dict | None = None
) -> dict:

    pre = pre or {}
    tau_global = float(pre.get("tau_global", 0.56))
    tau_lang = float(pre.get("tau_lang", 0.76))
    epsilon = float(pre.get("epsilon", 0.05)) if "epsilon" in pre else 0.05
    # Step 1: Add two opinions (BERT on MT English, mBERT on native)
    en_star = r_m.get("text_en", None)
    if en_star is None:
        try:
            en_star = predict_bert_plus_mt_best(text).get("text_en", None)
        except Exception:
            en_star = None
    try:
        r_b = predict_bert_en(en_star, do_translate=False) if en_star else None
    except Exception:
        r_b = None
    try:
        r_mb = predict_mbert(text)
    except Exception:
        r_mb = None
    # Step 2: Stability checks
    voters = []
    # X: Gate-1 XLM-R
    if r_x is not None:
        c_X = float(r_x.get("prob", 0.0))
        y_X = int(r_x.get("label", 0))
        w_bt_X = 1.0
        w_flip_X, flip_X = _run_flip_check("X", r_x, text, None, pre)
        w_stab_X = w_bt_X * w_flip_X
        c_X_stable = c_X * w_stab_X
        voters.append(("X", y_X, c_X_stable, r_x, w_stab_X))
    # M: Gate-2 MT→RoBERTa
    if r_m is not None:
        c_M = float(r_m.get("prob", 0.0))
        y_M = int(r_m.get("label", 0))
        w_bt_M = 1.0 if r_m.get("bt_ok") else 0.6
        w_flip_M, flip_M = _run_flip_check("M", r_m, text, r_m.get("text_en", None), pre)
        w_stab_M = w_bt_M * w_flip_M
        c_M_stable = c_M * w_stab_M
        voters.append(("M", y_M, c_M_stable, r_m, w_stab_M))
    # B: BERT on MT English
    if r_b is not None:
        c_B = float(r_b.get("prob", 0.0))
        y_B = int(r_b.get("label", 0))
        # For B, back-translation signal from r_m
        w_bt_B = 1.0 if r_m.get("bt_ok") else 0.6
        w_flip_B, flip_B = _run_flip_check("B", r_b, text, en_star, pre)
        w_stab_B = w_bt_B * w_flip_B
        c_B_stable = c_B * w_stab_B
        voters.append(("B", y_B, c_B_stable, r_b, w_stab_B))
    # m: mBERT on native
    if r_mb is not None:
        c_m = float(r_mb.get("prob", 0.0))
        y_m = int(r_mb.get("label", 0))
        w_bt_m = 1.0
        w_flip_m, flip_m = _run_flip_check("m", r_mb, text, None, pre)
        w_stab_m = w_bt_m * w_flip_m
        c_m_stable = c_m * w_stab_m
        voters.append(("m", y_m, c_m_stable, r_mb, w_stab_m))
    # Step 3: Build voting set: up to four (label, c_stable, who)
    voting_tuples = [(who, label, c_stable) for (who, label, c_stable, _, _) in voters]
    # Step 4: Apply decision rules
    
    # Rule 1: If all available c_stable < tau_global → abstain
    all_c_stable = [c_stable for (_, _, c_stable, _, _) in voters]
    if not all_c_stable or all(c < tau_global for c in all_c_stable):
        return {
            "undecided": True,
            "message": "All tie-breaker voters are below safety floor.",
            "tb_meta": {
                "voters": voting_tuples
            }
        }
    # Rule 2: If strict majority exists (> half the votes), accept that label
    label_counts = {}
    for (_, label, c_stable, _, _) in voters:
        label_counts[label] = label_counts.get(label, 0) + 1
    n_votes = len(voters)
    # Find majority
    majority_label = None
    for label, count in label_counts.items():
        if count > n_votes // 2:
            majority_label = label
            break
    if majority_label is not None:
        # Get max c_stable among votes for that label
        max_c = max(c_stable for (_, label, c_stable, _, _) in voters if label == majority_label)
        winner = next((rec for (_, label, c_stable, rec, _) in voters if label == majority_label and c_stable == max_c), None)
        if winner is not None:
            return {
                **winner,
                "used": "TieBreaker",
                "decision_level": "tie_breaker",
                "tb_meta": {
                    "voters": voting_tuples,
                    "winner": majority_label,
                    "conf": max_c
                }
            }
    # Rule 3: If 2–2 tie or generic tie, pick tied label with highest c_stable; else abstain
    label_max_c = {}
    for (_, label, c_stable, _, _) in voters:
        if label not in label_max_c or c_stable > label_max_c[label]:
            label_max_c[label] = c_stable
    if len(label_max_c) == 2 and all(list(label_counts.values())[0] == list(label_counts.values())[1] for _ in [0]):
        # 2-2 tie or equal votes
        # Pick tied label with highest c_stable
        best_label = max(label_max_c, key=lambda l: label_max_c[l])
        best_c = label_max_c[best_label]
        # If best_c < tau_global or top two c_stable within epsilon, abstain
        sorted_c = sorted(label_max_c.values(), reverse=True)
        if best_c < tau_global or (len(sorted_c) > 1 and abs(sorted_c[0] - sorted_c[1]) < epsilon):
            return {
                "undecided": True,
                "message": "Tie-breaker: tie, winner not strong/confident enough.",
                "tb_meta": {
                    "voters": voting_tuples,
                    "top_conf": best_c,
                    "top_margin": (sorted_c[0] - sorted_c[1]) if len(sorted_c) > 1 else 0.0,
                    "epsilon": epsilon
                }
            }
        # Accept
        winner = next((rec for (_, label, c_stable, rec, _) in voters if label == best_label and c_stable == best_c), None)
        if winner is not None:
            return {
                **winner,
                "used": "TieBreaker",
                "decision_level": "tie_breaker",
                "tb_meta": {
                    "voters": voting_tuples,
                    "winner": best_label,
                    "conf": best_c
                }
            }
    # Rule 4: If no majority but one label has more votes, accept it; else fall back to rule 3
    max_votes = max(label_counts.values())
    labels_with_max = [lbl for lbl, ct in label_counts.items() if ct == max_votes]
    if len(labels_with_max) == 1:
        winner_label = labels_with_max[0]
        winner_c = max(c_stable for (_, label, c_stable, _, _) in voters if label == winner_label)
        winner = next((rec for (_, label, c_stable, rec, _) in voters if label == winner_label and c_stable == winner_c), None)
        if winner is not None:
            return {
                **winner,
                "used": "TieBreaker",
                "decision_level": "tie_breaker",
                "tb_meta": {
                    "voters": voting_tuples,
                    "winner": winner_label,
                    "conf": winner_c
                }
            }
    # Rule 5: Weak-confidence trigger (minor_risk)
    minor_risk = False
    if dm_meta and dm_meta.get("minor_risk"):
        minor_risk = True
        # Only accept if winner's max c_stable >= tau_lang
        # Find winner as above
        winner_label = None
        winner_c = None
        if label_max_c:
            winner_label = max(label_max_c, key=lambda l: label_max_c[l])
            winner_c = label_max_c[winner_label]
        if winner_c is not None and winner_c >= tau_lang:
            winner = next((rec for (_, label, c_stable, rec, _) in voters if label == winner_label and c_stable == winner_c), None)
            if winner is not None:
                return {
                    **winner,
                    "used": "TieBreaker",
                    "decision_level": "tie_breaker",
                    "tb_meta": {
                        "voters": voting_tuples,
                        "winner": winner_label,
                        "conf": winner_c,
                        "minor_risk": True
                    }
                }
        # Otherwise, abstain
        return {
            "undecided": True,
            "message": "Tie-breaker: minor-risk, winner not above tau_lang.",
            "tb_meta": {
                "voters": voting_tuples,
                "minor_risk": True,
                "tau_lang": tau_lang,
                "winner_conf": winner_c
            }
        }
    # Default: abstain
    return {
        "undecided": True,
        "message": "Tie-breaker could not resolve; abstain.",
        "tb_meta": {
            "voters": voting_tuples
        }
    }



 # Decision module: accept, tie-break, or abstain based on confidences and thresholds
def _decision_module_gate12(text: str, r_x: dict, r_m: dict, pre: dict | None = None) -> dict:
    pre = pre or {}
    tau_lang   = float(pre.get("tau_lang", 0.76))      # accept bar (fallback)
    tau_global = float(pre.get("tau_global", 0.56))    # safety floor (fallback)
    delta_star = float(pre.get("delta_star", 0.11))    # margin threshold Δ* (fallback)

    # Extract labels (y) and effective confidences (c)
    y_X = int(r_x.get("label", 0))
    c_X = float(r_x.get("prob", 0.0))
    y_M = int(r_m.get("label", 0))
    c_M = float(r_m.get("prob", 0.0))
    agree = (y_X == y_M)
    Delta = abs(c_X - c_M)
    meta = {
        "agree": bool(agree),"delta": float(Delta),"delta_star": float(delta_star),
        "tau_lang": float(tau_lang),"tau_global": float(tau_global),"c_X": float(c_X),
        "c_M": float(c_M),"y_X": int(y_X),"y_M": int(y_M),
    }

    # 1) If they agree (close or not), accept shared label with higher confidence
    if agree:
        chosen = r_x if c_X >= c_M else r_m
        used = chosen.get("path", "Gate-1" if chosen is r_x else "Gate-2")
        chosen = {**chosen, "used": used, "decision_level": "decision_module"}
        return {"action": "accept", "chosen": chosen, "meta": meta}

    # From here on: they DISAGREE
    # 2) Both strong (≥ τ_lang) → tie‑breaker
    if (c_X >= tau_lang) and (c_M >= tau_lang):
        return {"action": "tie_break", "meta": meta}

    # 3) One strong (≥ τ_lang) and the other weak (< τ_lang):
    #    If not too close (Δ > Δ*), accept the strong; else tie‑breaker
    strong_X = (c_X >= tau_lang) and (c_M < tau_lang)
    strong_M = (c_M >= tau_lang) and (c_X < tau_lang)
    if strong_X or strong_M:
        if Delta > delta_star:
            chosen = r_x if strong_X else r_m
            used = chosen.get("path", "Gate-1" if chosen is r_x else "Gate-2")
            chosen = {**chosen, "used": used, "decision_level": "decision_module"}
            return {"action": "accept", "chosen": chosen, "meta": meta}
        else:
            return {"action": "tie_break", "meta": meta}

    # 4) Both weak but above safety floor (τ_global ≤ c < τ_lang) → tie‑breaker (minor‑risk)
    both_above_floor = (c_X >= tau_global) and (c_M >= tau_global)
    both_below_accept = (c_X < tau_lang) and (c_M < tau_lang)
    if both_above_floor and both_below_accept:
        return {"action": "tie_break", "meta": meta | {"minor_risk": True}}

    # 5) Below safety floor (max(c) < τ_global) → direct abstain
    if max(c_X, c_M) < tau_global:
        return {"action": "abstain", "meta": meta | {"high_risk": True}}

    # Default: tie‑breaker
    return {"action": "tie_break", "meta": meta}

 # Top-level router: run both gates, apply decision rules, return final verdict
def route_predict(text: str) -> dict:
    """
    Routing with Decision Module:
    1) Run Gate‑1 (XLM‑R) and Gate‑2 (best MT→RoBERTa).
    2) Apply Decision Module (Δ vs Δ*, τ_lang, τ_global) to accept / tie‑break / abstain.
    3) If tie‑breaker is requested, use new tie-breaker layer with all four opinions.
    """
    tried = []
    pre = prechecks(text)  # single precheck for thresholds & metadata

    # Gate‑1
    try:
        r_x = predict_xlmr(text)
        tried.append(r_x)
    except Exception:
        r_x = None

    # Gate‑2
    try:
        r_m = predict_roberta_plus_mt_best(text)
        tried.append(r_m)
    except Exception:
        r_m = None

    # If only one available, return it directly
    if r_x is None and r_m is None:
        raise RuntimeError("No XLM-RoBERTa or RoBERTa models are available.")
    if r_x is None:
        return r_m | {"used": r_m.get("path", "MT→RoBERTa"), "decision_level": "single_path"}
    if r_m is None:
        return r_x | {"used": r_x.get("path", "XLM-RoBERTa"), "decision_level": "single_path"}
    
    # Capture best MT branch details for UI even if not ultimately used
    mt_best_meta = None
    if r_m is not None:
        mt_best_meta = {
            "engine": r_m.get("engine"),
            "engine_version": r_m.get("engine_version"),
            "effective_prob": r_m.get("prob"),
            "bt_ok": r_m.get("bt_ok"),
            "qe_bucket": r_m.get("qe_bucket"),
        }

    # Decision Module (D)
    dm = _decision_module_gate12(text, r_x, r_m, pre)

    # Accept immediately
    if dm.get("action") == "accept":
        return dm["chosen"] | {"decision_meta": (dm.get("meta", {}) | {"mt_best": mt_best_meta})}

    # Direct abstain (below safety floor)
    if dm.get("action") == "abstain":
        return {
                "undecided": True,
                "all": tried,
                "message": "Below safety floor: both branches are too uncertain. Please review manually.",
                "decision_meta": dm.get("meta", {}) | {"mt_best": mt_best_meta}
                }

    # Tie-breaker requested → use _tie_breaker_layer
    tb = _tie_breaker_layer(text, r_x, r_m, pre, dm.get("meta"))
    # If accepted, return that
    if not tb.get("undecided"):
        return tb | {"decision_meta": dm.get("meta", {})}
    # If abstain/undecided, return abstain payload with tried models
    return {
        "undecided": True,
        "all": tried,
        "message": tb.get("message", "Tie-breaker could not resolve. Please review manually."),
        "tb_meta": tb.get("tb_meta"),
        "decision_meta": dm.get("meta", {}) | {"mt_best": mt_best_meta}
    }

 # Direct mBERT inference on the original text
def predict_mbert(text: str):
    logits = _logits(tokenizer_m, model_m, device_m, text)
    y, p, prob = _pred_from_logits(logits)
    return {"path": "mBERT", "label": y, "prob": p, "probs": prob, "logits": logits}

 # English BERT inference; translate first if requested
def predict_bert_en(text: str, do_translate=True):
    text_en = translate_to_en(text) if do_translate else text
    logits = _logits(tokenizer_e, model_e, device_e, text_en)
    y, p, prob = _pred_from_logits(logits)
    return {"path": "MT→BERT" if do_translate else "BERT(en)", "label": y, "prob": p,
            "probs": prob, "logits": logits, "text_en": text_en}

 # Simple router: prefer mBERT unless uncertain, then try BERT with translation
def auto_switch(text: str, tau=0.50, margin_thr=0.20, conf_triage=0.90,
                adv_use: bool = False, q0: float | None = None, cos0: float | None = None, r0: float | None = None, rt0: float | None = None):
    """
    1) Run mBERT → get entropy & margin.
    2) If confident enough → return mBERT.
    3) Else translate + run English BERT.
    4) (Optional) Use advanced signals (QE, LaBSE cosine, length penalty) to prefer BERT vs mBERT.
    5) If both confident but disagree → log for human triage.
    """
    # --- mBERT first ---
    m_out = predict_mbert(text)
    m_probs = _softmax(m_out["logits"])
    m_entropy = _entropy(m_probs)
    m_margin  = _top2_margin(m_probs)

    route_to_mt = (m_entropy > tau) or (m_margin < margin_thr)
    debug = {
        "m_entropy": float(m_entropy),
        "m_margin": float(m_margin),
        "routed_to_MT": bool(route_to_mt),
    }

    if not route_to_mt:
        return m_out | {"used": "mBERT", "debug": debug}

    # --- Translate + English BERT path ---
    b_out = predict_bert_en(text, do_translate=True)  # includes text_en inside
    text_en = b_out.get("text_en", None)

    # Optional advanced signals (computed only now, so no extra MT calls)
    qe = _comet_qe(text, b_out.get("text_en", text_en))
    cos = _labse_cosine(text, b_out.get("text_en", text_en))
    len_ratio, len_pen = _len_features(text, b_out.get("text_en", text_en))
    # round-trip EN → src cosine (higher = more consistent)
    try:
        src_lang, _ = detect_language(text)
    except Exception:
        src_lang = None
    rt_cos = _round_trip_cosine_from_en(text, text_en, src_lang)
    debug.update({
        "adv": {
            "qe": (None if qe is None else float(qe)),
            "labse_cosine": (None if cos is None else float(cos)),
            "len_ratio_en_over_src": float(len_ratio),
            "len_penalty_abs_log": float(len_pen),
            "round_trip_cosine": (None if rt_cos is None else float(rt_cos)),
        }
    })

    # disagreement triage: if both confident and disagree → write to feedback
    if (m_out["label"] != b_out["label"]) and (m_out["prob"] >= conf_triage) and (b_out["prob"] >= conf_triage):
        _append_feedback(text, m_out, b_out)

    # Decision:
    # Default: keep the higher-prob path
    best = b_out if b_out["prob"] >= m_out["prob"] else m_out
    used = "MT→BERT" if best is b_out else "mBERT"

    # If advanced routing enabled AND signals available, prefer BERT when QE high, cosine high, length reasonable, and round-trip consistent
    if adv_use and (qe is not None) and (cos is not None) and (q0 is not None) and (cos0 is not None) and (r0 is not None):
        prefer_bert = (qe >= q0) and (cos >= cos0) and (len_pen <= r0) and (rt_cos is None or rt_cos >= rt0)
        if prefer_bert:
            best, used = b_out, "MT→BERT"
        else:
            best, used = (b_out if b_out["prob"] >= m_out["prob"] else m_out), used

    return best | {"used": used, "debug": debug | {"b_prob": float(b_out["prob"]), "m_prob": float(m_out["prob"])}}

 # Log confident disagreements between models for later review
def _append_feedback(text, m_out, b_out, path=None):
    import pandas as pd, os
    from pathlib import Path
    row = {
        "input": text,
        "mb_label": int(m_out["label"]),  "mb_prob": float(m_out["prob"]),
        "bert_label": int(b_out["label"]), "bert_prob": float(b_out["prob"]),
        "note": "confident_disagreement",
    }
    path = path or FEEDBACK_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # ensure folder
    try:
        if os.path.exists(path):
            pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        else:
            pd.DataFrame([row]).to_csv(path, index=False)
    except Exception:
        pass


# ===== Add GCP client and language map======
PROJECT = "cogent-metric-470213-i8"   # or your project number
LOCATION = "global"


KEY = os.path.expanduser("~/Desktop/final_project/keys/translate-sa.json")
assert os.path.isfile(KEY), KEY  # sanity check

creds = service_account.Credentials.from_service_account_file(
    KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
client = translate.TranslationServiceClient(credentials=creds)
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"

 # Fetch supported languages from GCP and build name↔code maps
@st.cache_resource
def get_lang_maps():
    resp = client.get_supported_languages(parent=PARENT, display_language_code="en")
    name_to_code = {lang.display_name: lang.language_code for lang in resp.languages}
    code_to_name = {v: k for k, v in name_to_code.items()}
    return name_to_code, code_to_name
LANG_NAME2CODE, LANG_CODE2NAME = get_lang_maps()

 # Pretty-print a language code as 'Name (code)'
def _lang_full(code: str | None) -> str:
    """Return human-friendly language 'Name (code)' with case-insensitive fallback."""
    if not code:
        return "Unknown"
    if code in LANG_CODE2NAME:
        return f"{LANG_CODE2NAME[code]} ({code})"
    lc = code.lower()
    for k, v in LANG_CODE2NAME.items():
        if k.lower() == lc:
            return f"{v} ({k})"
    base = lc.split("-")[0]
    for k, v in LANG_CODE2NAME.items():
        if k.split("-")[0].lower() == base:
            return f"{v} ({code})"
    return code

 # Render figurative/slang flags as a short comma-separated string
def _flags_to_str(flags: dict | None) -> str:
    flags = flags or {}
    items = []
    if flags.get("idiom_or_sarcasm"):
        items.append("idiom/sarcasm")
    if flags.get("slang"):
        items.append("slang")
    return ", ".join(items) if items else "None"

 # Format a float as a percentage string with one decimal
def _nice_pct(x: float | None) -> str:
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return "—"

 # Summarize MT QE buckets per engine for display
def _fmt_qe_bins(qe: dict | None) -> str:
    qe = qe or {}
    g = qe.get("google", "—")
    n = qe.get("nllb", "—")
    return f"Google: {g}, NLLB: {n}"

# ==== MODEL CHECKPOINT CONFIG (NEW) ====
# Discover locally saved fine-tuned checkpoints in the project root (portable).
# You can override with env vars: XLMR_CKPT, ROBERTA_EN_CKPT, BERT_EN_CKPT, MBERT_CKPT.
try:
    ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
except Exception:
    ROOT = Path.cwd()

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _discover_local_ckpt(keys: list[str]) -> str | None:
    """Return the first subdirectory of ROOT whose name contains all key substrings
    (case-insensitive) and looks like a HF checkpoint (has config.json)."""
    try:
        for p in ROOT.iterdir():
            if not p.is_dir():
                continue
            name = p.name.lower()
            if all(k in name for k in keys) and (p / "config.json").exists():
                return str(p.resolve())
    except Exception:
        pass
    return None

MODEL_PATHS = {
    "xlmr":    os.getenv("XLMR_CKPT")       or _discover_local_ckpt(["xlm", "roberta", "model"]),
    "roberta": os.getenv("ROBERTA_EN_CKPT") or _discover_local_ckpt(["roberta", "model"]),
    "bert":    os.getenv("BERT_EN_CKPT")    or _discover_local_ckpt(["bert", "model"]),
    "mbert":   os.getenv("MBERT_CKPT")      or _discover_local_ckpt(["mbert", "model"]),
}

@st.cache_resource
def _load_mbert():
    try:
        ckpt = MODEL_PATHS.get("mbert")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-multilingual-uncased", num_labels=2
            ).to(_DEVICE)
        mdl.eval();  return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

@st.cache_resource
def _load_bert_en():
    try:
        ckpt = MODEL_PATHS.get("bert")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            ).to(_DEVICE)
        mdl.eval();  return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

@st.cache_resource
def _load_xlmr_cls():
    try:
        ckpt = MODEL_PATHS.get("xlmr")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "xlm-roberta-base", num_labels=2
            ).to(_DEVICE)
        mdl.eval();  return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

@st.cache_resource
def _load_roberta_en():
    try:
        ckpt = MODEL_PATHS.get("roberta")
        if ckpt and os.path.isdir(ckpt):
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt).to(_DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained("roberta-base")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            ).to(_DEVICE)
        mdl.eval();  return tok, mdl, _DEVICE
    except Exception:
        return None, None, _DEVICE

# Bind globals used across the pipeline (used later by predict_* functions)
try:
    tokenizer_m, model_m, device_m = _load_mbert()
    tokenizer_e, model_e, device_e = _load_bert_en()
    tokenizer_x, model_x, device_x = _load_xlmr_cls()
    tokenizer_r, model_r, device_r = _load_roberta_en()
except Exception:
    tokenizer_m = model_m = device_m = None
    tokenizer_e = model_e = device_e = None
    tokenizer_x = model_x = device_x = None
    tokenizer_r = model_r = device_r = None

# ==== CONSTANTS (unchanged) ====
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASETS_DIR = Path(os.getenv("DATA_DIR", ROOT / "Datasets")).expanduser().resolve()

MAIN_DATA_PATH = str(DATASETS_DIR / "main_data.csv")
FEEDBACK_PATH = str(DATASETS_DIR / "user_feedback.csv")
DEFAULT_COLUMNS = ["input", "label"]
FEEDBACK_COLUMNS = ["input", "label"]


# ==== LOAD MAIN DATA COLUMNS (unchanged logic) ====
if os.path.exists(MAIN_DATA_PATH):
    main_columns = pd.read_csv(MAIN_DATA_PATH, nrows=1).columns.tolist()
    for c in DEFAULT_COLUMNS:
        if c not in main_columns:
            main_columns.append(c)
else:
    main_columns = DEFAULT_COLUMNS



max_len = 512

# ==== PAGE SETUP & THEME (keep) ====
st.set_page_config(
    page_title="Fake/True News Detection",
    page_icon="🗞️",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": "Fake/True News Detection powered by BERT model"}
)


# Utility UI helpers:

def show_verdict_card(result: str, confidence: float):
    """
    'Card' that turns green for TRUE and red for FAKE,
    with a filled confidence bar inside.
    """
    pct = int(confidence * 100) #Converting into percentage
    slots = 20 # max Bar slots
    filled = max(0, min(slots, int(round(confidence * slots)))) # Slot should be between 0 and 20

    # Green bar for TRUE, Red bar for FAKE (emoji blocks)
    if result == "True":
        bar = "🟩" * filled + "⬜️" * (slots - filled)
        body = (
            "### ✅ TRUE NEWS\n\n"
            "**Confidence**\n"
            f"{pct}%\n\n"
            f"`{bar}{pct}%`\n\n"

        )
        st.success(body)  # for greenish box
    else:
        bar = "🟥" * filled + "⬜️" * (slots - filled)
        body = (
            "### 🚩 FAKE NEWS\n\n"
            "**Confidence**\n"
            f"{pct}%\n\n"
            f"`{bar}{pct}%`\n\n"
            
        )
        st.error(body)  # for reddish box


def status_analyzing():
    """Context manager box with status for analyzing."""
    return st.status("Analyzing…", expanded=False)

# Inserted contains_link_or_file helper after status_analyzing
def contains_link_or_file(text: str) -> bool:
    """
    Return True if the text looks like it contains a URL or a file reference.
    (Used to block links/files and keep inputs as plain text.)
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'(https?://|www\.|\.pdf|\.(jpg|jpeg|png|gif|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz|txt|mp3|mp4))',
        text,
        re.IGNORECASE
    ))

def safe_read_feedback(path: str) -> pd.DataFrame | None:
    """Read feedback CSV if present; return DataFrame or None so my webpage doesn't crash."""
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    except Exception:
        return None
    return None

# ==== SIDEBAR (keep language select + add session + quick tests) ====
st.sidebar.header("⚙️ Settings")

lang_list = sorted(LANG_NAME2CODE.keys())
selected_lang = st.sidebar.selectbox("Input Language", lang_list, index=lang_list.index("English"))
selected_lang_code = LANG_NAME2CODE[selected_lang]



# Tips (kept) + Manual download (kept)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tips**")
st.sidebar.caption("• Paste text only (no links/files)\n\n• Keep it under 1k characters\n\n• Use “Explain Prediction” to see top tokens.")
manual_path = Path("user_guide.docx")
st.sidebar.subheader("📄 User Manual")
if manual_path.exists():
    with manual_path.open("rb") as f:
        st.sidebar.download_button(
            label="Download User Manual",
            data=f.read(),
            file_name="user_guide.docx",
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.sidebar.warning(
        f"User manual not found at: `{manual_path}`\n\n"
        f"Working directory: `{Path.cwd()}`"
    )

# Session panel---------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Session")
st.sidebar.caption(f"Device: **{'cuda' if torch.cuda.is_available() else 'cpu'}**")
# model_choice = st.sidebar.selectbox("Choose model:", ["BERT (with translation)", "mBERT (without translation)"], index=0)
# st.session_state.model_choice = model_choice
st.sidebar.markdown("### Model mode")
mode = st.sidebar.radio("Choose model", ["Auto-switch", "XLM-RoBERTa","mBERT","roBERTa (+MT)","BERT (+MT)"])


if mode == "Auto-switch":
    with st.sidebar.expander("Auto-switch thresholds", expanded=False):
        tau = st.slider("Uncertainty threshold τ (entropy)", 0.0, 1.5, 0.50, 0.01)
        mth = st.slider("Low-margin threshold m (top1 - top2)", 0.0, 1.0, 0.20, 0.01)
        tri = st.slider("Disagreement triage conf.", 0.50, 0.99, 0.90, 0.01)

    with st.sidebar.expander("Advanced signals (optional)"):
        adv_use = st.checkbox("Use advanced signals for routing (QE + LaBSE + length)", value=False)
        if adv_use:
            q0  = st.slider("QE threshold q₀ (higher is better)",  -1.0, 1.0, 0.50, 0.01)
            cos0 = st.slider("LaBSE cosine threshold cos₀",         0.0, 1.0, 0.60, 0.01)
            r0  = st.slider("|log length-ratio| threshold r₀",      0.0, 1.5, 0.50, 0.01)
            rt0 = st.slider("Round-trip cosine threshold rt₀", 0.0, 1.0, 0.60, 0.01)
        else:
            q0 = cos0 = r0 = rt0 = None

elif mode == "BERT (+MT)":
    with st.sidebar.expander("Advanced signals BERT+MT (optional)"):
        bert_advise = st.checkbox(
            "Suggest switch to mBERT when BERT looks shaky",
            value=True,
            help="If BERT looks uncertain and mBERT is more confident, the app will offer to switch."
        )
        bert_min_conf = st.slider(
            "BERT min confidence to avoid suggestion",
            0.50, 0.99, 0.75, 0.01,
            help="If BERT's max probability is below this, it is considered 'shaky'."
        )
        mbert_min_adv = st.slider(
            "Min mBERT advantage Δprob",
            0.00, 0.50, 0.10, 0.01,
            help="mBERT must exceed BERT by at least this probability gap to suggest switching."
        )
else:
    # Defaults to avoid NameError when not in BERT mode
    bert_advise = False
    bert_min_conf = 0.75
    mbert_min_adv = 0.10
    
    

# Initialize shared aliases (use multilingual defaults for UI helpers / SHAP)
# Fall back to English BERT if mBERT failed to load.
if tokenizer_m is not None and model_m is not None:
    tokenizer, model, device = tokenizer_m, model_m, device_m
else:
    tokenizer, model, device = tokenizer_e, model_e, device_e

# ==== MAIN/MIDDLE HEADER ====
header = st.container(border=True)
with header:
    c1, c2 = st.columns([0.8, 0.2])   # c1 takes 80% width (for the title & caption). c2 takes 20% width (for a metric box).
    with c1:
        st.title("📰 Fake/True News Detection")
        st.caption("Paste a headline or short snippet. Language is auto-detected and translated to English for analysis if needed.")
    with c2:
        st.metric("Max Tokens", value="512")

 # Gather language, code-switch, QE, thresholds, and IDs before routing
def prechecks(text: str) -> dict:
    """
    Build a PrecheckReport:
    - language + confidence
    - code-switch ratio + alternate langs
    - MT QE bins per engine
    - figurative flags
    - engine allow-list/availability
    - thresholds (dynamic)
    - calibration ids per route
    - trace_id & text_hash
    """
    t0 = time.time()
    lang, lang_conf = detect_language(text)
    csr = _estimate_code_switch_ratio(text, lang)
    alt_langs = _detect_alt_langs(text, lang)
    qe_bins = _qe_bins_lookup(lang)
    fig_flags = _figurative_flags(text)
    engines = _engine_allowlist_and_versions()
    # dynamic thresholds from signals
    thr = compute_thresholds_dynamic(lang, lang_conf, csr, qe_bins, fig_flags)
    # calibration ids (placeholder IDs map to temps above)
    calib_ids = _calibration_ids_for_lang(lang)
    trace_id = str(uuid.uuid4())
    text_hash = _hash_text(text)
    pre = {
        "trace_id": trace_id, "text_hash": text_hash, "ts": time.time(), "lang": lang, "lang_conf": lang_conf, "code_switch_ratio": csr, "alt_langs": alt_langs, "mt_qe_bins": qe_bins,
        "figurative_flags": fig_flags,"engines": engines,"tau_lang": thr["tau_lang"],"delta_star": thr["delta_star"],"tau_global": thr["tau_global"],"calibration_ids": calib_ids,
    }
    # observability-safe log
    _obs_log("prechecks", { "trace_id": trace_id, "text_hash": text_hash,"lang": lang,"lang_conf": lang_conf, "code_switch_ratio": csr, "alt_langs": alt_langs, "mt_qe_bins": qe_bins,
        "figurative_flags": fig_flags,"engines": engines,"tau_lang": thr["tau_lang"],"delta_star": thr["delta_star"],"tau_global": thr["tau_global"],"calibration_ids": calib_ids,
        "ms": int((time.time() - t0) * 1000),
    })
    return pre


 # Heuristic check for presence of CJK characters in text
def is_cjk(text): # cjk = Chinese, Japanese, Korean
    """It used Unicode ranges to detect if the language contains any CJK characters."""
    return bool(re.search(
        r'[\u4E00-\u9FFF'
        r'\u3400-\u4DBF'
        r'\u20000-\u2A6DF'
        r'\u2A700-\u2B73F'
        r'\u2B740-\u2B81F'
        r'\u2B820-\u2CEAF'
        r'\uF900-\uFAFF'
        r'\u2F800-\u2FA1F'
        r'\u3040-\u30FF'
        r'\u31F0-\u31FF'
        r'\uAC00-\uD7AF'
        ']', text)
    )

 # Robust Chinese detector using langdetect plus Unicode count fallback
def robust_is_chinese(text):
    try:
        if len(text) > 8:
            lang = detect(text)  # Try language detection (langdetect library)
            if lang in ('zh-cn', 'zh-tw'):
                return True
    except Exception:
        pass
    return len(re.findall(r'[\u4e00-\u9fff]', text)) >= 3


 # Normalize a language code to lowercase with hyphens
def _norm_code(c: str | None) -> str | None:
    return c.lower().replace("_", "-") if c else None

 # Detect language with GCP and return (code, confidence) with CJK fallbacks
def detect_language(text: str):
    """Return (language_code, confidence) or (None, None) on failure."""
    txt = str(text or "").strip()
    if not txt:
        return None, None
    try:
        resp = client.detect_language(request={"parent": PARENT, "content": txt})
        # choose the language with highest confidence
        if hasattr(resp, "languages") and resp.languages:
            cand = max(resp.languages, key=lambda x: getattr(x, "confidence", 0.0) or 0.0)
            lang = getattr(cand, "language_code", None)   # keep GCP casing, e.g., 'en', 'zh-CN'
            conf = float(getattr(cand, "confidence", 0.0) or 0.0)
        else:
            lang, conf = None, None
    except Exception:
        lang, conf = None, None
    # Heuristic fallbacks for CJK when GCP detection is unavailable
    if not lang:
        if robust_is_chinese(txt):
            return "zh-CN", 1.0
        if is_cjk(txt):
            return "zh", 1.0
    return lang, conf

 # Quick heuristic: mostly ASCII and has a vowel → likely English
def looks_english(s: str) -> bool:
    if not s:
        return True
    ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
    return ascii_ratio >= 0.98 and any(v in s.lower() for v in "aeiou")


 # Translate text using Google Cloud Translation API
def gcp_translate(text: str, target: str = "en", source: str | None = None) -> str:
    req = {
        "parent": PARENT,
        "contents": [text],
        "mime_type": "text/plain",
        "target_language_code": target,
    }
    if source:
        req["source_language_code"] = source
    resp = client.translate_text(request=req)
    return resp.translations[0].translated_text

 # Translate text using NLLB if available, else fall back to GCP
def nllb_translate(text: str, target: str = "en", source: str | None = None) -> str:
    """
    NLLB-200 translation via HuggingFace. Falls back to GCP or identity on failure.
    """
    if not text or not text.strip():
        return text
    tok, mdl, dev = _load_nllb()
    if tok is None or mdl is None:
        # Fallback to GCP if NLLB not available
        try:
            return gcp_translate(text, target=target, source=source)
        except Exception:
            return text
    try:
        src_code = source
        if not src_code:
            try:
                src_code, _ = detect_language(text)
            except Exception:
                src_code = None
        src_tag = _nllb_tag_from_iso(src_code)
        tgt_tag = _nllb_tag_from_iso(target)

        tok.src_lang = src_tag
        enc = tok(str(text), return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        gen = mdl.generate(**enc, forced_bos_token_id=tok.lang_code_to_id[tgt_tag], max_length=512)
        out = tok.batch_decode(gen, skip_special_tokens=True)
        return out[0] if out else text
    except Exception:
        try:
            return gcp_translate(text, target=target, source=source)
        except Exception:
            return text


CJK_RE = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\u31F0-\u31FF\uAC00-\uD7AF]')
SYMBOLS_ONLY_RE = re.compile(r'[\W\d_]+', flags=re.UNICODE)  # Matches strings made entirely of: Non-word characters (\W → punctuation, emoji, etc.), Digits (\d), Underscores (_).

 # Count the number of CJK characters in a string
def count_cjk(s: str) -> int:  # Counts how many CJK characters are in a string
    return len(CJK_RE.findall(s))


 # Filter out inputs that are empty, symbols-only, or too short to analyze
def is_garbage(text):
    if not isinstance(text, str): # If the input is not a string, it’s garbage
        return True
    txt = unicodedata.normalize('NFKC', text).strip() # Normalize all text
    if not txt: # Empty after trimming → garbage.
        return True
    if SYMBOLS_ONLY_RE.fullmatch(txt):  # If the whole thing is only symbols/numbers, → garbage.
        return True
    cjk_chars = count_cjk(txt) # Check if it is cjk which has more than 10 char. 
    if cjk_chars >= 10:
        return False
    if len(txt) < 10 or len(txt.split()) < 2:  # Too short (<10 characters) or fewer than 2 words → garbage.
        return True
    common = {"the","and","is","was","to","for","of","with","on","in"} # If text has none of the most common English words and is shorter than 40 chars, it’s garbage.
    lacks_common = sum(1 for w in txt.lower().split() if w in common) < 1
    if lacks_common and len(txt) < 40:
        return True
    return False

 # Canonicalize text for hashing and duplicate checks
def _canon(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)              # Unicode normalize
    s = s.replace("\r", " ").replace("\n", " ")       # single-line
    s = re.sub(r"\s+", " ", s).strip()                # collapse spaces
    return s

 # Stable SHA-256 hash of canonicalized text
def _hash_text(s: str) -> str:
    try:
        return hashlib.sha256(_canon(s).encode("utf-8")).hexdigest()
    except Exception:
        return "na"

 # Map a QE bucket label to a numeric weight
def _qe_weight_from_bin(bin_name: str) -> float:
    """Map QE bucket → weight."""
    b = (bin_name or "").strip().lower()
    if b == "good":  return 1.0
    if b == "ok":    return 0.7
    if b == "poor":  return 0.3
    return 0.7  # default to OK

 # Baseline thresholds per language group
def _get_thresholds(lang_code: str) -> dict:
    """
    Coarse, language-group defaults.
    These are BASELINES; dynamic adjustments will refine them per item.
    """
    lc = (lang_code or "").lower()
    high = {"en", "es", "de", "hi"}
    low  = {"vi", "id", "sw", "bn"}
    if lc in high:
        return {"tau_lang": 0.75, "delta_star": 0.10, "tau_global": 0.55}
    if lc in low:
        return {"tau_lang": 0.78, "delta_star": 0.12, "tau_global": 0.57}
    # unknown/mid → safe defaults
    return {"tau_lang": 0.76, "delta_star": 0.11, "tau_global": 0.56}

 # Adjust thresholds based on risk signals like lang confidence and code-switch
def compute_thresholds_dynamic(
    lang_code: str,
    lang_conf: float | None,
    code_switch_ratio: float,
    qe_bins: dict,
    figurative_flags: dict,
) -> dict:
    """
    Adaptive thresholds when language is unknown or risky.
    Signals used: low lang-ID confidence, high code-switch, poor MT QE, figurative/slang.
    Returns: {'tau_lang', 'delta_star', 'tau_global}
    """
    base = _get_thresholds(lang_code)  # coarse defaults per language/group
    tau_lang   = float(base["tau_lang"])     # accept bar for a single gate
    delta_star = float(base["delta_star"])   # “close” margin between c_X and c_M
    tau_global = float(base["tau_global"])   # safety floor

    # ---- Risk from signals ----
    risk = 0.0

    # 1) Language-ID confidence (lower → riskier)
    if lang_conf is not None:
        if lang_conf < 0.40:
            risk += 1.0
        elif lang_conf < 0.60:
            risk += 0.5

    # 2) Code-switch ratio (higher → riskier)
    if code_switch_ratio >= 0.50:
        risk += 1.0
    elif code_switch_ratio >= 0.30:
        risk += 0.5

    # 3) MT QE bins (if both engines weak → riskier)
    w_g = _qe_weight_from_bin((qe_bins or {}).get("google", "OK"))
    w_n = _qe_weight_from_bin((qe_bins or {}).get("nllb",  "OK"))
    best_qe = max(w_g, w_n)
    if best_qe < 0.50:
        risk += 1.0
    elif best_qe < 0.70:
        risk += 0.5

    # 4) Figurative / slang
    if (figurative_flags or {}).get("idiom_or_sarcasm"):
        risk += 0.5
    if (figurative_flags or {}).get("slang"):
        risk += 0.5

    # ---- Convert risk → threshold adjustments ----
    tau_lang   = tau_lang   + 0.02 * risk       # stricter accept bar
    delta_star = delta_star + 0.04 * risk       # widen “too close” → tie-breaker more often
    tau_global = tau_global + 0.01 * risk       # raise safety floor a bit

    # ---- Clamp to safe ranges ----
    tau_lang   = float(min(0.90, max(0.60, tau_lang)))
    delta_star = float(min(0.20, max(0.05, delta_star)))
    tau_global = float(min(0.65, max(0.50, tau_global)))

    return {"tau_lang": tau_lang, "delta_star": delta_star, "tau_global": tau_global}

# --- Unicode-aware sentence splitter (includes Devanagari danda: \u0964, double danda: \u0965) ---
SENT_END_RE = re.compile(r"[\.\!\?\n;।॥]+")

# --- Script utilities for code-switch estimation ---
 # Identify the writing script of a character
def _script_of(ch: str) -> str:
    cp = ord(ch)
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
        return "latin"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
        return "cjk"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    return "other"

 # Count letters per script to estimate code-switching
def _script_mix_counts(text: str) -> dict:
    counts = {"latin": 0, "devanagari": 0, "cjk": 0, "arabic": 0, "other": 0}
    for ch in text:
        # count only letters from common scripts; skip digits/whitespace
        if ch.isalpha():
            counts[_script_of(ch)] += 1
    return counts

_PRIMARY_SCRIPT_BY_LANG = {
    # latin
    "en": "latin", "fr": "latin", "de": "latin", "es": "latin", "pt": "latin", "it": "latin",
    # devanagari
    "ne": "devanagari", "hi": "devanagari", "mr": "devanagari", "sa": "devanagari",
    # cjk
    "zh": "cjk", "zh-cn": "cjk", "zh-tw": "cjk", "ja": "cjk", "ko": "cjk",
}

_ALT_LANG_BY_SCRIPT = {"latin": "en", "devanagari": "ne", "cjk": "zh", "arabic": "ar"}

 # Pick primary script based on language code or observed text
def _primary_script_for_lang(lang_code: str | None, text: str) -> str:
    lc = (lang_code or "").lower()
    if lc in _PRIMARY_SCRIPT_BY_LANG:
        return _PRIMARY_SCRIPT_BY_LANG[lc]
    # Fallback: pick dominant script in text
    cnt = _script_mix_counts(text)
    return max(cnt, key=cnt.get)

 # Proportion of letters not in the primary script
def _code_switch_ratio_by_script(text: str, primary_lang: str | None) -> float:
    """Proportion of letters from scripts other than the primary script."""
    cnt = _script_mix_counts(text)
    total_letters = sum(cnt.values())
    if total_letters == 0:
        return 0.0
    primary_script = _primary_script_for_lang(primary_lang, text)
    non_primary = total_letters - cnt.get(primary_script, 0)
    return float(min(1.0, max(0.0, non_primary / total_letters)))

 # Hybrid chunk+script estimator for code-switch ratio
def _estimate_code_switch_ratio(text: str, primary_lang: str | None) -> float:
    txt = _canon(text)
    if not txt:
        return 0.0

    # --- (1) Chunk-level ratio ---
    chunks = SENT_END_RE.split(txt)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 4]
    try:
        primary = (primary_lang or "").lower()
    except Exception:
        primary = ""
    total_tokens = 0
    switched_tokens = 0
    for ch in chunks:
        try:
            lang, _ = detect_language(ch)
            lang = (lang or "").lower()
        except Exception:
            lang = primary
        tokens = max(1, len(ch.split()))
        total_tokens += tokens
        # treat chunks as switched only if their script differs
        primary_script = _primary_script_for_lang(primary, txt)
        if lang:
            if _primary_script_for_lang(lang, txt) != primary_script:
                switched_tokens += tokens
    ratio_chunk = float(min(1.0, switched_tokens / max(1, total_tokens)))

    # --- (2) Script-mix ratio ---
    ratio_script = _code_switch_ratio_by_script(txt, primary_lang)
    return float(max(ratio_chunk, ratio_script))

 # Detect up to two alternate languages present in the text
def _detect_alt_langs(text: str, primary_lang: str | None) -> list[str]:
    """
    Detect up to two additional languages:
    - chunk-level lang-ID over sentence-like pieces (Unicode-aware)
    - script-mix hints (e.g., Latin inside Devanagari → add 'en')
    """
    txt = _canon(text)
    if not txt:
        return []

    chunks = SENT_END_RE.split(txt)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 4]

    primary = (primary_lang or "").lower()
    counts = {}
    for ch in chunks:
        try:
            lang, _ = detect_language(ch)
            lang = (lang or "").lower()
        except Exception:
            continue
        if primary and lang == primary:
            continue
        if not lang:
            continue
        counts[lang] = counts.get(lang, 0) + 1

    # Script-based hints
    cnt = _script_mix_counts(txt)
    total = sum(cnt.values())
    if total > 0:
        primary_script = _primary_script_for_lang(primary, txt)
        # consider a script 'present' if it contributes at least 10% of letters
        for script, n in cnt.items():
            if script == primary_script:
                continue
            if n / total >= 0.10:
                lang_hint = _ALT_LANG_BY_SCRIPT.get(script)
                if lang_hint and lang_hint != primary:
                    counts[lang_hint] = counts.get(lang_hint, 0) + 1

    alt = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [code for code, _ in alt[:2]]

 # Light heuristics to flag sarcasm or slang
def _figurative_flags(text: str) -> dict:
    """
    Heuristic figurative/sarcasm/slang indicators.
    This is intentionally conservative; it only raises light flags.
    """
    t = (_canon(text) or "").lower()
    idiom_or_sarcasm = any(p in t for p in [
        "/s", "yeah right", "as if", "totally not", "sure, jan", "what a joke",
        "obviously...", "sarcasm", "irony", "satire"
    ])
    slang = any(p in t.split() for p in ["lol", "lmao", "rofl", "omg", "wtf", "bruh", "fr", "rofl","hahaha"])
    return {"idiom_or_sarcasm": bool(idiom_or_sarcasm), "slang": bool(slang)}

 # Coarse per-language MT quality buckets for Google and NLLB
def _qe_bins_lookup(lang_code: str) -> dict:
    """
    Coarse per-language QE bins per engine (Good/OK/Poor).
    Based on thesis rationale: Google stronger for high-resource; NLLB stronger for low-resource/code-switch.
    """
    lc = (lang_code or "").lower()
    high = {"en", "es", "de", "hi"}
    low  = {"vi", "id", "sw", "bn"}
    if lc in high:
        return {"google": "Good", "nllb": "OK"}
    if lc in low:
        return {"google": "OK", "nllb": "Good"}
    # unknown → neutral
    return {"google": "OK", "nllb": "OK"}

 # Which MT engines are allowed and their version/availability
def _engine_allowlist_and_versions() -> dict:
    """Allow-list + version tags + availability for observability."""
    return {
        "google": {"allowed": True, "version": "gcp-translate-v3", "available": True},
        "nllb":   {"allowed": True, "version": "facebook/nllb-200-distilled-600M", "available": _nllb_available()},
    }

 # Pick calibration IDs per route; swap to real IDs when available
def _calibration_ids_for_lang(lang_code: str) -> dict:
    """
    Placeholder calibration IDs per route.
    If you later fit temperature per language, swap these IDs to real ones.
    """
    return {
        "xlmr":   f"temp-xlmr-default",
        "roberta":f"temp-roberta-default",
        "mbert":  f"temp-mbert-default",
        "bert":   f"temp-bert-default",
    }
# --- Calibration temperatures (IDs -> numeric T) and helpers ---
CALIBRATION_TEMPS = {
    # Placeholders (1.00 = no scaling until you fit temps)
    "temp-xlmr-default":    1.00,
    "temp-roberta-default": 1.00,
    "temp-mbert-default":   1.00,
    "temp-bert-default":    1.00,
}

 # Look up temperature scalar for a given route
def _get_temperature(route_key: str, calib_ids: dict | None) -> float:
    """Return temperature for a route key (e.g., 'xlmr') using provided calibration_ids."""
    try:
        if not calib_ids:
            return 1.0
        key = (calib_ids.get(route_key) or "").strip()
        return float(CALIBRATION_TEMPS.get(key, 1.0))
    except Exception:
        return 1.0

 # Apply temperature scaling to logits
def _apply_temperature(logits, T: float):
    """Divide logits by temperature T (T>0) prior to softmax."""
    try:
        if T is None or T <= 0:
            T = 1.0
    except Exception:
        T = 1.0
    return (logits / T)


 # Append a PII-safe event to the observability JSONL log
def _obs_log(event_type: str, payload: dict):
    """
    PII-safe JSONL logging to Datasets/obs_log.jsonl
    (no raw text; use text_hash & metadata only)
    """
    try:
        path = Path(FEEDBACK_PATH).parent / "obs_log.jsonl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event_type, **payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass
 # Render engine availability as checkmarks for the UI
def _fmt_engine_availability(engines: dict | None) -> str:
    e = engines or {}
    def mark(name: str) -> str:
        meta = (e.get(name) or {})
        ok = bool(meta.get("allowed", True) and meta.get("available", True))
        return f"{name.capitalize()}: {'✅' if ok else '❌'}"
    return f"{mark('google')}, {mark('nllb')}"

 # Streamlit UI: render a compact analysis card with key signals
def _render_complete_analysis(pre: dict, out: dict, text: str,
                            dm_meta: dict | None = None,
                            tb_meta: dict | None = None):
    """
    Render a compact, user-friendly summary card + optional technical JSON.
    • Shows: Label, Confidence, Model, Language, Code-switch ratio, Figurative flags,
    MT QE bins and (only when used) the Best MT chosen line, thresholds and trace_id.
    • For explicit XLM-R path, also show entropy (H) and top-2 margin (m).
    """
    # ---- Guards ----
    if not isinstance(out, dict) or out is None:
        st.warning("No result payload to render.")
        return
    if not isinstance(pre, dict) or pre is None:
        pre = {}
        # NEW: explicit RoBERTa (+MT) report
    if isinstance(out, dict) and out.get("explicit_mode") == "roBERTa (+MT)":
        st.subheader("Complete analysis")

        left, right = st.columns([0.48, 0.52])

        with left:
            st.markdown("#### Result summary")
            st.markdown(f"**Label:** {_label_name(out.get('label', 0))}")
            st.markdown(f"**Confidence:** {float(out.get('prob', 0.0)):.3f}")
            st.markdown("**Model chosen:** RoBERTa (+MT)")
            st.markdown(f"**Language:** {_lang_full(pre.get('lang'))}")
            st.markdown(f"**Code-switch ratio:** {_nice_pct(pre.get('code_switch_ratio'))}")
            st.markdown(f"**Figurative flags:** {_flags_to_str(pre.get('figurative_flags'))}")
            st.markdown(f"**MT QE bins:** {_fmt_qe_bins(pre.get('mt_qe_bins'))}")

            eng = out.get("engine") or "—"
            ver = out.get("engine_version") or "—"
            eff = float(out.get("prob", 0.0))
            bt_ok = bool(out.get("bt_ok"))
            len_ok = (out.get("len_penalty_abs_log") is not None) and (float(out.get("len_penalty_abs_log")) <= float(MT_LEN_L0))

            st.markdown("#### MT engine (selected)")
            st.markdown(
                f"- **Engine:** {eng} (ver: {ver})  \n"
                f"- **Effective score S_e:** {eff:.3f}  \n"
                f"- **Stability:** round-trip {'✅' if bt_ok else '❌'} ; length {'✅' if len_ok else '❌'}  \n"
                f"- **Weights:** w_QE={float(out.get('w_QE',1.0)):.2f}, w_BT={float(out.get('w_BT',1.0)):.2f}, "
                f"w_sem={float(out.get('w_sem',1.0)):.2f}, w_len={float(out.get('w_len',1.0)):.2f}"
            )

            st.markdown(
                f"_τ_lang / Δ* / τ_global_: {pre.get('tau_lang',0.0):.2f} / {pre.get('delta_star',0.0):.2f} / {pre.get('tau_global',0.0):.2f}  \n"
                f"**trace_id:** `{pre.get('trace_id')}`"
            )

        with right:
            st.markdown("#### Engines evaluated")
            rows = []
            for c in (out.get("scores_all") or []):
                len_ok_c = (c.get("len_penalty_abs_log") is not None) and (float(c.get("len_penalty_abs_log")) <= float(MT_LEN_L0))
                rows.append({
                    "Engine": c.get("engine") or "—",
                    "QE": c.get("qe_bucket") or "—",
                    "w_QE": round(float(c.get("w_QE", 1.0)), 2),
                    "BT stable": "✅" if c.get("bt_ok") else "❌",
                    "w_BT": round(float(c.get("w_BT", 1.0)), 2),
                    "w_sem": round(float(c.get("w_sem", 1.0)), 2),
                    "w_len": round(float(c.get("w_len", 1.0)), 2),
                    "Round-trip cosine": None if c.get("round_trip_cosine") is None else round(float(c.get("round_trip_cosine")), 3),
                    "|log len|": round(float(c.get("len_penalty_abs_log") or 0.0), 3),
                    "Len ok": "✅" if len_ok_c else "❌",
                    "RoBERTa p": round(float(c.get("roberta_prob", 0.0)), 3),
                    "Score S_e": round(float(c.get("score", 0.0)), 3),
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=min(360, 60 + 28*len(rows)))
            else:
                st.caption("No alternative engines were evaluated.")

        _obs_log("prediction", {
            "trace_id": pre.get("trace_id"),
            "text_hash": pre.get("text_hash"),
            "path": out.get("path"),
            "label": int(out.get("label", 0)),
            "prob": float(out.get("prob", 0.0)),
            "lang": pre.get("lang"),
            "code_switch_ratio": float(pre.get("code_switch_ratio") or 0.0),
            "figurative_flags": pre.get("figurative_flags"),
            "tau_lang": float(pre.get("tau_lang", 0.0)),
            "delta_star": float(pre.get("delta_star", 0.0)),
            "tau_global": float(pre.get("tau_global", 0.0)),
            "used": "roBERTa (+MT) explicit",
            "engine_selected": out.get("engine"),
        })
        return
    # ---- Core fields ----
    try:
        label_name = _label_name(int(out.get("label", 0)))
    except Exception:
        label_name = _label_name(0)

    try:
        conf = float(out.get("prob", 0.0))
    except Exception:
        conf = 0.0

    model_used = out.get("used", out.get("path", "—"))
    lang_full = _lang_full(pre.get("lang"))
    csr_str = _nice_pct(pre.get("code_switch_ratio"))
    flags_str = _flags_to_str(pre.get("figurative_flags"))

    tau_lang = pre.get("tau_lang")
    delta_star = pre.get("delta_star")
    tau_global = pre.get("tau_global")
    delta_val = (dm_meta or {}).get("delta") if isinstance(dm_meta, dict) else None

    # ---- MT used? (only show Best MT when actually used) ----
    path_lower = str(out.get("path", "")).lower()
    engine = out.get("engine")
    mt_used = (( "mt" in path_lower) and ("bert" in path_lower or "roberta" in path_lower)) or (engine in {"google", "nllb"})

    # Coerce probs to a plain list for any JSON we may show later
    _probs_raw = out.get("probs", None)
    if isinstance(_probs_raw, np.ndarray):
        _probs_list = _probs_raw.flatten().tolist()
    elif isinstance(_probs_raw, (list, tuple)):
        _probs_list = list(_probs_raw)
    else:
        _probs_list = []

    # ---- Summary Card ----
    card = st.container(border=True)
    with card:
        st.subheader("Complete analysis")
        colL, colR = st.columns([0.56, 0.44])

        with colL:
            st.markdown(f"**Label:** {label_name}")
            st.markdown(f"**Confidence:** {conf:.3f}")
            st.markdown(f"**Model chosen:** {model_used}")
            st.markdown(f"**Language:** {lang_full}")
            st.markdown(f"**Code-switch ratio:** {csr_str}")
            st.markdown(f"**Figurative flags:** {flags_str}")

            # XLM-R extras (uncertainty)
            if str(out.get("path", "")).lower().startswith("xlm-r"):
                H = out.get("entropy")
                m = out.get("margin")
                if H is not None and m is not None:
                    st.markdown("**Uncertainty:**")
                    st.markdown(f"**Entropy H** = {float(H):.4f}") 
                    st.markdown(f"**Top-2 margin m** = {float(m):.4f}")

        with colR:
            # QE bins always relevant (language awareness)
            st.markdown(f"**MT QE bins:** {_fmt_qe_bins(pre.get('mt_qe_bins'))}")

            # Best MT chosen (ONLY when MT path actually used)
            if mt_used:
                base_model_name = "RoBERTa" if "roberta" in path_lower else ("BERT" if "bert" in path_lower else "Classifier")
                raw_p = out.get("roberta_prob", out.get("bert_prob", None))
                bt_ok = out.get("bt_ok")
                qe_bucket = out.get("qe_bucket")
                ver = out.get("engine_version", "—")
                temp = float(out.get("temperature", 1.0))
                raw_display = float(raw_p) if raw_p is not None else conf
                bt_str = "stable" if bt_ok else "not stable"
                st.markdown(
                    f"**Best MT chosen:** {engine or '—'} (ver: {ver}) "
                    f"T={temp:.2f} ; {base_model_name} p={raw_display:.3f} ; "
                    f"effective={conf:.3f} ; BT {bt_str} ; QE={qe_bucket or '—'}"
                )

            # Thresholds and Δ (if present)
            try:
                st.markdown(
                    f"**τ_lang / Δ* / τ_global:** "
                    f"{float(tau_lang or 0.0):.2f} / {float(delta_star or 0.0):.2f} / {float(tau_global or 0.0):.2f}"
                )
            except Exception:
                st.markdown("**τ_lang / Δ* / τ_global:** —")

            if isinstance(delta_val, (int, float)):
                st.markdown(f"**Δ (c_X − c_M):** {float(delta_val):.3f}")

            st.markdown(f"**trace_id:** `{pre.get('trace_id', '—')}`")

        # Optional: Technical JSON (collapsed)
        with st.expander("Technical details (JSON)", expanded=False):
            pre_view = {k: pre.get(k) for k in [
                "lang", "code_switch_ratio", "mt_qe_bins", "figurative_flags",
                "tau_lang", "delta_star", "tau_global", "trace_id"
            ]}
            out_view = dict(out)
            out_view["probs"] = _probs_list
            st.json({
                "prechecks": pre_view,
                "result": out_view,
                "decision_meta": dm_meta,
                "tie_breaker": tb_meta,
            })

    # ---- Observability log (PII-safe) ----
    try:
        _obs_log("prediction", {
            "trace_id": pre.get("trace_id"),
            "text_hash": pre.get("text_hash"),
            "path": out.get("path"),
            "used": out.get("used", out.get("path")),
            "label": int(out.get("label", 0)),
            "prob": float(conf),
            "lang": pre.get("lang"),
            "code_switch_ratio": float(pre.get("code_switch_ratio") or 0.0),
            "figurative_flags": pre.get("figurative_flags"),
            "tau_lang": float(pre.get("tau_lang", 0.0)),
            "delta_star": float(pre.get("delta_star", 0.0)),
            "tau_global": float(pre.get("tau_global", 0.0)),
            "engine": engine,
        })
    except Exception:
        pass

# ==== EXPLAIN TAB UTILITIES ==================================================

def _token_importance_leave_one_out(tokenizer, model, device, text: str, target_label: int,
                                    max_tokens: int = 80) -> tuple[list[str], list[float]]:
    """Simple explanation: leave-one-out occlusion.
    For up to max_tokens tokens:
    - mask (or delete) the token,
    - re-run model,
    - importance = base_prob(target) − new_prob(target).
    """
    if tokenizer is None or model is None or not str(text).strip():
        return [], []
    model.eval()

    encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    if len(tokens) <= 2:
        return tokens, [0.0] * len(tokens)

    with torch.no_grad():
        base_logits = model(**{k: v.to(_DEVICE) for k, v in encoded.items()}).logits
        base_p = float(torch.softmax(base_logits, dim=-1)[0, target_label].item())

    mask_tok = getattr(tokenizer, "mask_token", None)
    idxs = list(range(len(tokens)))[:max_tokens]
    importances = []

    for i in idxs:
        toks_mod = list(tokens)
        # skip specials
        if toks_mod[i] in {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}:
            importances.append(0.0)
            continue
        if mask_tok:
            toks_mod[i] = mask_tok
        else:
            del toks_mod[i]
        try:
            text_mod = tokenizer.convert_tokens_to_string(toks_mod)
            enc_mod = tokenizer(text_mod, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits_mod = model(**{k: v.to(_DEVICE) for k, v in enc_mod.items()}).logits
                p_mod = float(torch.softmax(logits_mod, dim=-1)[0, target_label].item())
            imp = max(0.0, base_p - p_mod)
        except Exception:
            imp = 0.0
        importances.append(imp)

    mx = max(importances) if importances else 0.0
    scores = [(x / mx) if mx > 1e-12 else 0.0 for x in importances]
    disp_tokens = [t.replace("Ġ", "").replace("▁", "") for t in tokens[:max_tokens]]
    return disp_tokens, scores


def _render_token_heatmap(tokens: list[str], scores: list[float]):
    """Render tokens as a simple heatmap (darker = more influence)."""
    if not tokens:
        st.info("No tokens to explain.")
        return
    chunks = []
    for t, s in zip(tokens, scores):
        alpha = min(1.0, max(0.0, s))
        bg = f"rgba(255, 99, 71, {alpha:.2f})"  # tomato
        safe = t.replace("<", "&lt;").replace(">", "&gt;")
        chunks.append(f"<span style='background:{bg}; padding:2px 4px; margin:2px; border-radius:4px; display:inline-block'>{safe}</span>")
    st.markdown(" ".join(chunks), unsafe_allow_html=True)



# ==== SHAP-based explanation helpers ====
import math

_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_'\-]*$")

def _clean_vis_token(t: str) -> str:
    t = (t or "").replace("Ġ", "").replace("▁", "").strip()
    t = re.sub(r"^\W+|\W+$", "", t)
    return t

def _is_word_like(t: str) -> bool:
    if not t or t.isspace():
        return False
    if t.lower() in _SK_STOP:
        return False
    if not _WORD_RE.match(t):
        return False
    return True

def _predict_proba_texts(texts, tokenizer, model, device, label_index: int = 1):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    return probs[:, label_index].detach().cpu().numpy()

def _render_token_heat_table_single(title: str, rows: list[tuple[str, float]], color: str):
    # rows: list of (token, value) already signed for the direction we're showing
    if not rows:
        st.info("No token contributions available.")
        return
    # normalize by max |value|
    mx = max(abs(v) for _, v in rows) or 1.0
    st.markdown(f"### {title}")
    # Build HTML table
    html = [
        "<div style='border:1px solid #eee;border-radius:6px;overflow:hidden'>",
        "<table style='width:100%;border-collapse:collapse;font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif'>",
        "<thead><tr>",
        "<th style='text-align:left;padding:10px;border-bottom:1px solid #eee'>token</th>",
        f"<th style='text-align:right;padding:10px;border-bottom:1px solid #eee'>{'contribution_to_true' if color=='green' else 'contribution_to_fake'}</th>",
        "</tr></thead><tbody>"
    ]
    for tok, val in rows:
        alpha = min(1.0, max(0.0, abs(val) / mx))
        bg = f"rgba(0,128,0,{alpha:.2f})" if color == "green" else f"rgba(220,20,60,{alpha:.2f})"
        html.append(
            "<tr>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #f3f3f3'>{tok}</td>"
            f"<td style='padding:8px 10px;border-bottom:1px solid #f3f3f3;text-align:right;background:{bg};color:#000'>{val:+.6f}</td>"
            "</tr>"
        )
    html.append("</tbody></table></div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


def _render_explain_for_last():
    last_out = st.session_state.get("last_out")
    last_text = st.session_state.get("last_input", "")
    if not isinstance(last_out, dict):
        st.info("Run a prediction first.")
        return

    # Determine model/tokenizer/text actually used for the prediction
    path = (last_out.get("used") or last_out.get("path") or "").lower()
    label = int(last_out.get("label", 0))  # 0=True, 1=Fake

    if "xlm-r" in path:
        tok, mdl, dev = tokenizer_x, model_x, device_x
        text_for_model = last_text
    elif "mbert" in path and "mt" not in path:
        tok, mdl, dev = tokenizer_m, model_m, device_m
        text_for_model = last_text
    elif "roberta" in path:
        tok, mdl, dev = tokenizer_r, model_r, device_r
        text_for_model = last_out.get("text_en", last_text)
    elif "bert" in path:
        tok, mdl, dev = tokenizer_e, model_e, device_e
        text_for_model = last_out.get("text_en", last_text)
    else:
        # Fallback
        tok, mdl, dev = (tokenizer_m or tokenizer_x), (model_m or model_x), (device_m or device_x)
        text_for_model = last_text

    if tok is None or mdl is None or not text_for_model:
        st.warning("Explanation not available: model/tokenizer not loaded.")
        return

    # Try SHAP text explainer
    tokens, shap_vals = [], []
    try:
        masker = shap.maskers.Text(tok.tokenize)
        def f(texts):
            return _predict_proba_texts(texts, tok, mdl, dev, label_index=1)
        explainer = shap.Explainer(f, masker)
        ex = explainer([text_for_model])
        vals = ex.values[0]
        toks = ex.data[0]
        tokens = [ _clean_vis_token(t) for t in toks ]
        shap_vals = [ float(v) for v in vals ]
    except Exception:
        # Fallback to leave-one-out occlusion
        toks, scores = _token_importance_leave_one_out(tok, mdl, dev, text_for_model, target_label=label)
        tokens = [ _clean_vis_token(t) for t in toks ]
        # Map occlusion (positive = lowers prob of target) to SHAP-like sign w.r.t Fake prob
        # We approximate by assuming target_label==1 → positive pushes to Fake; label==0 → negative pushes to True.
        shap_vals = [ float(s if label==1 else -s) for s in scores ]

    # Filter tokens: words only (no stop words / whitespace / punctuation)
    clean = []
    for t, v in zip(tokens, shap_vals):
        if _is_word_like(t):
            clean.append((t, v))

    if not clean:
        st.info("No informative tokens found.")
        return

    # Select top 10 in the predicted direction
    if label == 0:  # TRUE → take most negative values
        direction = sorted(clean, key=lambda x: x[1])[:10]
        _render_token_heat_table_single("Tokens pushing this TRUE verdict", direction, color="green")
        st.caption("More negative values = stronger push towards **True** (green).")
    else:  # FAKE → take most positive values
        direction = sorted(clean, key=lambda x: x[1], reverse=True)[:10]
        _render_token_heat_table_single("Tokens pushing this FAKE verdict", direction, color="red")
        st.caption("More positive values = stronger push towards **Fake** (red).")
    
def prechecks(text: str) -> dict:
    """
    Runs all prechecks and returns a PrecheckReport dict:
    {
        'trace_id', 'lang', 'lang_conf', 'code_switch_ratio',
        'mt_qe_bins': {'google','nllb'},
        'figurative_flags': {'idiom_or_sarcasm', 'slang'},
        'tau_lang', 'delta_star', 'tau_global',
        'calibration_ids', 'engines': {...}
    }
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())

    # Language ID
    lang_code, lang_conf = detect_language(text)
    lang_code = (lang_code or "").lower()

    # Code-switch
    csr = _estimate_code_switch_ratio(text, lang_code)
    alt_langs = _detect_alt_langs(text, lang_code)

    # QE bins (lookup level)
    qe_bins = _qe_bins_lookup(lang_code)

    # Figurative flags
    flags = _figurative_flags(text)

    # Thresholds (dynamic)
    th = compute_thresholds_dynamic(
        lang_code=lang_code,
        lang_conf=(lang_conf or 0.0),
        code_switch_ratio=csr,
        qe_bins=qe_bins,
        figurative_flags=flags,
    )

    # Calibration & engine meta
    calib_ids = _calibration_ids_for_lang(lang_code)
    eng_meta  = _engine_allowlist_and_versions()

    report = {
        "trace_id": trace_id,
        "lang": lang_code,
        "lang_conf": float(lang_conf or 0.0),
        "code_switch_ratio": float(csr),
        "mt_qe_bins": qe_bins,
        "figurative_flags": flags,
        "tau_lang": float(th["tau_lang"]),
        "delta_star": float(th["delta_star"]),
        "tau_global": float(th["tau_global"]),
        "calibration_ids": calib_ids,   # IDs only; real temps applied elsewhere
        "engines": eng_meta,
        "alt_langs": alt_langs
    }

    # Observability-safe log
    try:
        _obs_log("prechecks", {
            "alt_langs": alt_langs,
            "trace_id": trace_id,
            "text_hash": _hash_text(text),
            "lang": report["lang"],
            "lang_conf": report["lang_conf"],
            "code_switch_ratio": report["code_switch_ratio"],
            "mt_qe_bins": report["mt_qe_bins"],
            "figurative_flags": report["figurative_flags"],
            "tau_lang": report["tau_lang"],
            "delta_star": report["delta_star"],
            "tau_global": report["tau_global"],
            "calibration_ids": report["calibration_ids"],
            "engines": report["engines"],
            "duration_ms": int((time.time() - t0) * 1000),
        })
    except Exception:
        pass

    return report

for key, default in {
    "prediction_done": False,
    "show_feedback_form": False,
    "last_input": "",
    "last_result": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def start_correction():
    """Switches on the feedback form in the app by setting its session state flag."""
    st.session_state.show_feedback_form = True

# ==== PREDICT ACTION ====
# This section drives prediction for all modes and renders a neat summary.
if 'text_input' in locals() and 'predict_button' in locals() and predict_button:
    # Reset UI flags
    st.session_state.prediction_done = False
    st.session_state.show_feedback_form = False
    st.session_state.pop("switch_offer", None)

    # Basic input validation
    if not text_input or text_input.strip() == "":
        st.error("❗ Please enter a news headline or article text.")
    elif contains_link_or_file(text_input):
        st.error("❗ Please enter only news text, not URLs or file names.")
    elif is_garbage(text_input):
        st.error("❗ Please paste a valid news headline or article.")
    else:
        # Language gate against sidebar selection
        detected_lang_code, detect_conf = detect_language(text_input)
        if not detected_lang_code or detected_lang_code not in LANG_CODE2NAME:
            st.error("❗ Language could not be detected or is not supported.")
        elif detected_lang_code != selected_lang_code:
            det_name = LANG_CODE2NAME.get(detected_lang_code, "Unknown")
            st.error(
                f"❗ Please paste the news in the chosen language ({selected_lang}), "
                f"or change your selection. Detected: {det_name}."
            )
        else:
            # Safe token length check for the currently bound tokenizer (mBERT used for estimate)
            encoded = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=max_len)
            if encoded['input_ids'].shape[1] > max_len:
                st.error(f"❗ Text is too long! Please keep your input under {max_len} tokens.")
            else:
                with status_analyzing() as s:
                    # 1) Run Prechecks once (thresholds, QE bins, flags, trace_id, etc.)
                    pre = prechecks(text_input)

                    # 2) Route by mode
                    out = None
                    dm_meta = None
                    tb_meta = None

                    try:
                        if mode == "Auto-switch":
                            # Full pipeline: Gate-1, Gate-2, Decision Module, Tie-breaker
                            routed = route_predict(text_input)
                            # route_predict returns final record or abstain dict; preserve meta if present
                            dm_meta = routed.get("decision_meta") if isinstance(routed, dict) else None
                            tb_meta = routed.get("tb_meta") if isinstance(routed, dict) else None
                            out = routed
                        elif mode == "XLM-RoBERTa":
                            out = predict_xlmr(text_input)
                            if isinstance(out, dict):
                                out["explicit_mode"] = "XLM-RoBERTa"   # <-- tag as explicit XLM-R
                            dm_meta = tb_meta = None
                        elif mode == "roBERTa (+MT)":
                            out = predict_roberta_plus_mt_best(text_input)
                            out = out | {"used": out.get("path", "MT→RoBERTa"), "decision_level": "single_path"}
                        elif mode == "mBERT":
                            out = predict_mbert(text_input)
                            out = out | {"used": out.get("path", "mBERT"), "decision_level": "single_path"}
                        elif mode == "BERT (+MT)":
                            out = predict_bert_plus_mt_best(text_input)
                            out = out | {"used": out.get("path", "BERT(+MT)"), "decision_level": "single_path"}
                        else:
                            st.error("Unsupported mode.")
                            out = None
                    except Exception as e:
                        st.error(f"❗ Error during processing: {e}")
                        out = None
                    try:
                        s.update(label="Analyzing completed", state="complete")
                    except Exception:
                        pass

                # Neat “Complete analysis” panel
                with st.container(border=True):
                    st.subheader("Complete analysis")
                    _render_complete_analysis(pre, out, text_input, dm_meta, tb_meta)

                    # 3) Render result
                    if isinstance(out, dict) and ("label" in out) and ("prob" in out):
                        # Verdict card
                        show_verdict_card(_label_name(int(out.get("label", 0))), float(out.get("prob", 0.0)))

                        st.session_state.prediction_done = True
                        st.session_state.last_input = text_input
                        st.session_state.last_result = _label_name(int(out.get("label", 0)))
                    else:
                        # Abstain / undecided payload
                        st.warning(out.get("message", "Analysis complete but no decisive prediction. Please review manually.") if isinstance(out, dict) else "Analysis complete but no result.")
                        # Show technical info if available
                        if isinstance(out, dict):
                            with st.expander("Technical details (JSON)", expanded=False):
                                st.json(out)
# ===================== BOTTOM-OF-FILE UI (runs after all defs) =====================
# Recreate tabs now that all functions are defined.
tab_predict, tab_explain, tab_history, tab_help = st.tabs(["Predict", "Explain", "History", "Help"])

with tab_predict:
    # ---- Input card ----
    card = st.container(border=True)
    with card:
        text_input = st.text_area(
            "Input Text",
            placeholder="e.g., NASA confirms successful landing of the Artemis mission...",
            height=170,
            max_chars=1000,
            key="news_text_area",
        )

        # Action row (buttons + token counter)
        act1, act2, act3 = st.columns([0.22, 0.26, 0.52])
        with act1:
            predict_button = st.button("🔍 Predict", use_container_width=True)
        with act2:
            st.button(
                "🧹 Clear",
                help="Clear the text box",
                on_click=lambda: st.session_state.update({"news_text_area": ""}),
                use_container_width=True,
            )
        with act3:
            if text_input:
                est_tokens = len(tokenizer.encode(text_input, truncation=True, max_length=512))
                st.caption(f"Estimated tokens: **{est_tokens}** / 512")

    # ---- FULL-WIDTH inference & rendering (outside columns) ----
    if predict_button:
        if contains_link_or_file(text_input):
            st.error("Links/files are not allowed. Please paste plain text.")
        elif not text_input.strip():
            st.warning("Please enter some text.")
        else:
            # Use spinner only; no status cards.
            with st.spinner("Analyzing…"):
                try:
                    pre = prechecks(text_input)

                    if mode == "Auto-switch":
                        out = route_predict(text_input)
                        dm_meta = out.get("decision_meta") if isinstance(out, dict) else None
                        tb_meta = out.get("tb_meta") if isinstance(out, dict) else None
                    elif mode == "XLM-RoBERTa":
                        out = predict_xlmr(text_input)
                        if isinstance(out, dict):
                            out["explicit_mode"] = "XLM-RoBERTa"   # <-- tag as explicit XLM-R
                        dm_meta = tb_meta = None
                    elif mode == "mBERT":
                        out = predict_mbert(text_input); dm_meta = tb_meta = None
                    elif mode == "roBERTa (+MT)":
                        out = predict_roberta_plus_mt_best(text_input); dm_meta = tb_meta = None
                    else:  # "BERT (+MT)"
                        out = predict_bert_plus_mt_best(text_input); dm_meta = tb_meta = None

                except Exception as e:
                    st.error(f"❗ Error during processing: {e}")
                    out = None
                    dm_meta = tb_meta = None

            if isinstance(out, dict):
                # 1) Verdict card FIRST (full width, above the report)
                show_verdict_card(_label_name(out.get("label", 0)), float(out.get("prob", 0.0)))
                # 2) Detailed report beneath
                _render_complete_analysis(pre, out, text_input, dm_meta=dm_meta, tb_meta=tb_meta)
                
                # — Persist latest prediction for Explain tab —
                st.session_state["prediction_done"] = True
                st.session_state["last_input"] = text_input
                st.session_state["last_pre"] = pre
                st.session_state["last_out"] = out

with tab_explain:
    if not st.session_state.get("prediction_done"):
        st.info("Run a prediction on the Predict tab to see the explanation.")
    else:
        _render_explain_for_last()

with tab_history:
    df_fb = safe_read_feedback(FEEDBACK_PATH)
    if df_fb is not None and len(df_fb) > 0:
        st.dataframe(df_fb, use_container_width=True)
    else:
        st.caption("No feedback yet.")

with tab_help:
    st.markdown("""
**How to use**
1. Paste a short headline or snippet.
2. Choose a model mode in the sidebar.
3. Click **Predict**. You’ll see the verdict card and a detailed report.
""")