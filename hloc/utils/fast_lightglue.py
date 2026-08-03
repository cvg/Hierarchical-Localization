"""LightGlue の CrossBlock を Apple Silicon (MPS) で高速化するパッチ（opt-in）。

upstream の `CrossBlock.forward` は SDPA を使う高速パスを
`device.type == "cuda"` で明示的にガードしている。MPS でも SDPA は
利用可能かつ大幅に速いのに、fallback の einsum 経路に落ちてしまう。
fallback は sim 行列を実体化したうえで softmax を 2 回、さらに
`transpose(-2, -1).contiguous()` で全要素コピーするため、SDPA の
数倍のメモリ帯域を消費する。

同じ block の SelfBlock は Attention モジュール経由なので既に MPS で
SDPA を使っており、CrossBlock だけが取り残されている状態。

HLOC_FAST_LIGHTGLUE=0 で無効化できる。
"""

import os
from typing import List, Optional

import torch

from .. import logger

_patched = False


def _cross_forward(
    self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    qk0, qk1 = self.map_(self.to_qk, x0, x1)
    v0, v1 = self.map_(self.to_v, x0, x1)
    qk0, qk1, v0, v1 = map(
        lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
        (qk0, qk1, v0, v1),
    )
    # upstream: `self.flash is not None and qk0.device.type == "cuda"`
    if self.flash is not None and qk0.device.type in ("cuda", "mps"):
        m0 = self.flash(qk0, qk1, v1, mask)
        m1 = self.flash(
            qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
        )
    else:
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
        if mask is not None:
            sim = sim.masked_fill(~mask, -float("inf"))
        attn01 = torch.nn.functional.softmax(sim, dim=-1)
        attn10 = torch.nn.functional.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()
    m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
    m0, m1 = self.map_(self.to_out, m0, m1)
    x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
    x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
    return x0, x1


def tune_conf_for_device(conf: dict, device: str) -> dict:
    """MPS では point pruning を切る。

    枝刈りは層ごとに keypoint 数を変えるため、MPS ではその都度シェイプの
    異なるカーネルが生成される。gather のコストと合わせて、削減できる
    計算量よりオーバーヘッドの方が大きくなる（実測 574ms -> 245ms/pair）。
    枝刈りは元々近似なので、無効化してもマッチは減らない。

    HLOC_LIGHTGLUE_PRUNING=1 で upstream の挙動に戻せる。
    """
    if device != "mps" or os.environ.get("HLOC_LIGHTGLUE_PRUNING") == "1":
        return conf
    if conf.get("width_confidence", -1) > 0:
        conf = dict(conf)
        conf["width_confidence"] = -1
        logger.info(
            "LightGlue: disabling point pruning on MPS (kernel recompilation "
            "outweighs the savings); set HLOC_LIGHTGLUE_PRUNING=1 to keep it."
        )
    return conf


def patch_lightglue(device: str) -> bool:
    """CrossBlock の SDPA 高速パスを MPS でも有効にする。適用したら True。"""
    global _patched
    if _patched:
        return True
    if os.environ.get("HLOC_FAST_LIGHTGLUE") == "0":
        return False
    if device != "mps":
        return False

    try:
        from lightglue.lightglue import CrossBlock
    except ImportError:
        return False

    CrossBlock.forward = _cross_forward
    _patched = True
    logger.info("LightGlue: enabled the SDPA cross-attention fast path on MPS.")
    return True
