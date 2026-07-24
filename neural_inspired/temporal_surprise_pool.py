"""Predictive-coding-inspired surprise-weighted temporal pooling.

Optional brain-inspired add-on (see ``neural_inspired/README.md``). Depends only
on ``torch`` and is reachable *only* through the model builder's explicit
``temporal_pooling="surprise"`` branch; the baseline model never imports it.
"""

import torch
import torch.nn as nn


class TemporalSurprisePool(nn.Module):
    """Surprise-weighted pooling over the T per-frame embeddings.

    Motivation (predictive coding; Rao & Ballard, 1999): the temporal *gist* is
    the mean frame embedding (the "prediction"); each frame's **prediction
    error** — how far it deviates from that gist — signals how informative it is.
    We pool frames with weights proportional to their surprise, and gate the whole
    correction with a scalar ``alpha`` initialized to 0 so that **at step 0 this
    module is bit-for-bit mean pooling** (strictly-no-regression start, mirroring
    :class:`models.mobileviclip_small.TemporalAttentionPool`).

    Teacher-free; training and inference are identical. Drop-in for the temporal
    pooling contract used by the vision encoder: input ``[B, T, C]`` per-frame
    embeddings, output ``[B, C]``.

    **Gradient-flow requirement — why ``gamma`` starts nonzero.** ``alpha``'s own
    gradient is ``dL/d(out) . (pooled - mean)``. If ``gamma`` also started at 0,
    the softmax would be uniform, ``pooled`` would *equal* ``mean``, so
    ``(pooled - mean) == 0`` and **every** parameter's gradient (``alpha``,
    ``gamma``, ``key``) would be exactly 0 — the init is a zero-gradient fixed
    point and the module can never leave mean pooling (verified by autograd; see
    ``experiments/pilot_10k_results.md``). Initializing ``gamma`` to a nonzero
    value makes ``pooled != mean`` at init, giving ``alpha`` a live gradient so
    the gate can open. Crucially this keeps the no-regression start intact:
    ``alpha=0`` still forces ``output == mean`` at step 0 regardless of ``gamma``.

    "Did it train?" tell (for the V100/10k plumbing screen): ``alpha`` leaves 0
    (primary gate), and ``gamma`` / ``key`` leave their init — i.e. the module
    actually started weighting frames by surprise instead of averaging them.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Learned "error space": which feature directions count toward surprise.
        self.key = nn.Linear(dim, dim, bias=False)
        # Surprise sharpness. MUST be nonzero: gamma=0 makes the weights uniform,
        # which zeroes the gradient to *every* parameter (see class docstring).
        # gamma>0 up-weights high-surprise frames (the predictive-coding prior).
        self.gamma = nn.Parameter(torch.ones(1))
        # Gate on the surprise correction. 0 -> output == mean at init (no
        # regression); its gradient is live because gamma!=0 makes pooled!=mean.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] per-frame embeddings; returns [B, C].
        mean = x.mean(1)                                  # [B, C]  prediction / gist
        err = self.key(x - mean.unsqueeze(1))             # [B, T, C]  prediction error
        surprise = err.pow(2).mean(-1)                    # [B, T]  per-frame error energy
        w = torch.softmax(self.gamma * surprise, dim=1)   # [B, T]  surprise weights
        pooled = torch.einsum("bt,btc->bc", w, x)         # [B, C]  surprise-weighted sum
        # alpha=0 at init -> exactly mean pooling; grows only if it helps.
        return mean + self.alpha * (pooled - mean)
