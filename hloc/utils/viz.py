"""
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


def cm_RdGn(x: npt.ArrayLike) -> np.ndarray:
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x_arr = np.clip(x, 0, 1)[..., None] * 2
    c = x_arr * np.array([[0, 1.0, 0]]) + (2 - x_arr) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(
    imgs: Sequence[np.ndarray | torch.Tensor],
    titles: Sequence[str] | None = None,
    cmaps: str | list[str] | tuple[str, ...] = "gray",
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
    figsize: float = 4.5,
) -> None:
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    fig_size = (sum(ratios) * figsize, figsize)
    fig, axs = plt.subplots(
        1, n, figsize=fig_size, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        axs = [axs]
    for i, (img, ax) in enumerate(zip(imgs, axs, strict=True)):
        ax.imshow(img, cmap=plt.get_cmap(cmaps[i]))
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(
    kpts: Sequence[np.ndarray],
    colors: str | tuple[float, ...] | list[Any] = "lime",
    ps: float = 4,
) -> None:
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, (k, c) in zip(axes, zip(kpts, colors, strict=True), strict=False):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    color: str | tuple[float, ...] | Sequence[Any] | None = None,
    lw: float = 1.5,
    ps: float = 4,
    indices: tuple[int, int] = (0, 1),
    a: float = 1.0,
) -> None:
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    colors: Sequence[Any]
    if color is None:
        colors = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        colors = [color] * len(kpts0)
    else:
        colors = color

    if lw > 0:
        # transform the points into the figure coordinate system
        for i in range(len(kpts0)):
            fig.add_artist(
                matplotlib.patches.ConnectionPatch(
                    xyA=(kpts0[i, 0], kpts0[i, 1]),
                    coordsA=ax0.transData,
                    xyB=(kpts1[i, 0], kpts1[i, 1]),
                    coordsB=ax1.transData,
                    zorder=1,
                    color=colors[i],
                    linewidth=lw,
                    alpha=a,
                )
            )

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=colors, s=ps)


def add_text(
    idx: int,
    text: str,
    pos: tuple[float, float] = (0.01, 0.99),
    fs: float = 15,
    color: str | tuple[float, ...] = "w",
    lcolor: str | tuple[float, ...] | None = "k",
    lwidth: float = 2,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax = plt.gcf().axes[idx]
    t = ax.text(
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path: Path | str, **kw: Any) -> None:
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
