"""Level 12D: OPTIONAL embeddings layer for the RAG (semantic and CROSS-LANGUAGE retrieval).

Thin layer over `knowledge.py`: TF-IDF stays the DEFAULT. A toggle in config.env
(`EMBEDDINGS=0` by default = off) enables a HYBRID search (lexical TF-IDF + semantic).
If the model is not installed/available, it degrades HONESTLY: it falls back to TF-IDF and warns
(never breaks). With OFF, the RAG behaves exactly as before.

Local and free model (Apple Silicon, native MLX, no cloud and no cost): multilingual-e5-small
(~241 MB on disk), which retrieves well ACROSS languages (fixes the documented limit: query in ES vs
document in EN). OPTIONAL dependency (`requirements-embeddings.txt`); if missing, it degrades.

Lightweight context untouched: the layer only re-ranks/adds candidates; it still returns the top-k with citation.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/multilingual-e5-small-mlx"
BATCH = 64                          # batch size when embedding (bounded memory)
# The multilingual e5 cosine has a high BASELINE (~0.6 even between unrelated texts); we
# rescale by subtracting this floor, so "0.6 irrelevant" -> 0 and "0.85 relevant" -> real signal.
EMB_FLOOR = 0.70
W_LEX, W_SEM = 0.5, 0.5            # hybrid weights (lexical / semantic)

_CFG_ENV = None                    # config.env cache
_MODEL = None                      # cache: (model, tokenizer) | "FAILED"
_WARNED = False                    # warn about degradation only once


def _config_env():
    global _CFG_ENV
    if _CFG_ENV is None:
        d = {}
        try:
            for ln in (ROOT / "config.env").read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, _, v = ln.partition("=")
                    d[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
        _CFG_ENV = d
    return _CFG_ENV


def _flag(cfg, key, default=""):
    """Value of a key: FIRST the passed cfg (for tests/callers), then os.environ, then config.env."""
    if cfg is not None and key in cfg:
        return str(cfg.get(key) or "")
    val = os.environ.get(key)
    if val is None:
        val = _config_env().get(key, default)
    return str(val or "")


def enabled(cfg=None):
    """Is the embeddings layer on? EMBEDDINGS toggle (default 0 = off)."""
    return _flag(cfg, "EMBEDDINGS", "0").strip().lower() in ("1", "true", "yes", "on")


def model(cfg=None):
    return _flag(cfg, "EMBEDDINGS_MODEL").strip() or DEFAULT_MODEL


def _load(cfg=None):
    """Load (cached) the model. None if the lib/model are missing -> honest degradation (warns once)."""
    global _MODEL, _WARNED
    if _MODEL is not None:
        return _MODEL if _MODEL != "FAILED" else None
    try:
        from mlx_embeddings.utils import load
        _MODEL = load(model(cfg))
    except Exception:  # noqa: BLE001 - lib not installed / model unavailable / no network
        _MODEL = "FAILED"
        if not _WARNED:
            import sys
            print("embeddings: EMBEDDINGS active but the model is not available "
                  "(install requirements-embeddings.txt) -> using TF-IDF only (honest degradation).",
                  file=sys.stderr)
            _WARNED = True
        return None
    return _MODEL


def available(cfg=None):
    """Can the semantic layer be used? (lib + model loadable). Never raises."""
    return _load(cfg) is not None


def _embed(texts, prefix, cfg=None):
    mt = _load(cfg)
    if mt is None or not texts:
        return None
    model_obj, tokenizer = mt
    try:
        import mlx_embeddings
        import numpy as np
        vecs = []
        for i in range(0, len(texts), BATCH):
            batch = [prefix + (t or "") for t in texts[i:i + BATCH]]
            out = mlx_embeddings.generate(model_obj, tokenizer, texts=batch)
            vecs.append(np.array(out.text_embeds, dtype="float32"))
        m = np.concatenate(vecs, axis=0)
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (m / norm).tolist()                 # L2-normalized -> cosine = dot product
    except Exception:  # noqa: BLE001 - any model failure -> None (caller falls back to TF-IDF)
        return None


def embed_passages(texts, cfg=None):
    """Embeddings (L2-normalized) of PASSAGES (e5 prefix 'passage: '). None if no model."""
    return _embed(texts, "passage: ", cfg)


def embed_query(text, cfg=None):
    """Embedding (L2-normalized) of the QUERY (e5 prefix 'query: '). None if no model."""
    r = _embed([text], "query: ", cfg)
    return r[0] if r else None


def rescale_semantic(cosine):
    """Rescale an e5 cosine by subtracting the baseline: <EMB_FLOOR -> 0; EMB_FLOOR..1 -> 0..1."""
    return max(0.0, (cosine - EMB_FLOOR) / (1.0 - EMB_FLOOR))
