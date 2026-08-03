"""SuperGlue の GPU 向け高速化パッチ（opt-in）。

- attention を einsum 実装から `scaled_dot_product_attention` に差し替える。
  数値は MPS 上で bit 一致、CPU では einsum の方が速いため GPU のみ適用。
- 任意で GNN を float16 autocast で実行する（sinkhorn は log 空間なので fp32 のまま）。

HLOC_FAST_SUPERGLUE=0 で無効化、HLOC_SUPERGLUE_FP16=1 で fp16 を有効化。
"""

import os

import torch
import torch.nn.functional as F

from .. import logger

_patched = False


def _attention_sdpa(query, key, value):
    """query/key/value: (B, D, H, N) -> ((B, D, H, N), None)

    SDPA の既定スケールは 1/sqrt(D) で、元実装の `/ dim**.5` と一致する。
    元実装は attention 確率も返すが SuperGlue 内では捨てられるので None を返す。
    """
    q, k, v = (x.permute(0, 2, 3, 1) for x in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 3, 1, 2), None


def patch_superglue(device: str) -> bool:
    """SuperGlue の attention を SDPA に差し替える。適用したら True。"""
    global _patched
    if _patched:
        return True
    if os.environ.get("HLOC_FAST_SUPERGLUE") == "0":
        return False
    if device not in ("mps", "cuda"):
        return False  # CPU では einsum の方が速い

    try:
        from SuperGluePretrainedNetwork.models import superglue as sg_module
    except ImportError:
        logger.debug("SuperGlue module not importable, skipping fast path.")
        return False

    sg_module.attention = _attention_sdpa
    _patched = True
    logger.info("SuperGlue: using scaled_dot_product_attention (fast path).")
    return True


def use_fp16() -> bool:
    return os.environ.get("HLOC_SUPERGLUE_FP16") == "1"


class autocast_if_enabled:
    """fp16 が有効なときだけ autocast する context manager。"""

    def __init__(self, device: str):
        self.enabled = use_fp16() and device in ("mps", "cuda")
        self.device = device

    def __enter__(self):
        if self.enabled:
            self.ctx = torch.autocast(device_type=self.device, dtype=torch.float16)
            self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            return self.ctx.__exit__(*exc)
        return False
