"""Opt-in **brain-inspired add-ons** for MobileViCLIP.

Everything here is an *optional* research add-on layered on top of the baseline
model. Each component is reachable only through an explicit config knob (e.g.
``model.vision_encoder.temporal_pooling = "surprise"``) and is off by default.
The baseline never imports this package; the model builder imports a class from
here lazily, inside the matching config branch.

Design contract (so these can be dropped wholesale if their measured gain is
minor — see ``README.md``):
  * torch-only, self-contained, no dependency on the baseline model modules;
  * every component starts **bit-for-bit equal to the baseline** it replaces
    (zero-gated), so turning it on can never regress step 0.
"""

from .temporal_surprise_pool import TemporalSurprisePool

__all__ = ["TemporalSurprisePool"]
