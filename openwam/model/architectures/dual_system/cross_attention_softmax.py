"""DualSystem cross-attention with a joint-softmax bridge (``variant: cross_attention_softmax``).

Identical to :mod:`joint_cross_attn` (video DiT runs to completion; per-layer bridge features feed a separate
ActionDiT; ``detach_bridge`` / ``bridge_kv_dim`` / ``action_video_attention_mode`` behave the same) except for the
bridge attention inside every action block: its query attends to the action tokens AND the video bridge tokens in
ONE softmax, instead of a softmax over the video tokens alone. An action query can therefore keep part of its
attention on action rather than always taking a convex combination of video values; this isolates "shared softmax"
from the other differences between the cross- and self-attention (MoT) families.

Extra parameters per block: action-side K/V projections and a key RMSNorm for the bridge (``cross_attn.k_a``,
``cross_attn.v_a``, ``cross_attn.norm_k_a``).
"""

from __future__ import annotations

from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture, _cross_attn_options
from openwam.model.architectures.registry import register_architecture


@register_architecture(
    "dual_system_cross_attn_softmax",
    status="supported",
    note="DualSystem cross-attention whose bridge query attends to [action ; video] keys in one softmax.",
    framework="dual_system",
    variant="cross_attention_softmax",
    options_from_cfg=_cross_attn_options,
)
class DualSystemCrossAttnSoftmaxArchitecture(DualSystemCrossAttnArchitecture):
    _FORCE_BRIDGE_JOINT_SOFTMAX = True


__all__ = ["DualSystemCrossAttnSoftmaxArchitecture"]
