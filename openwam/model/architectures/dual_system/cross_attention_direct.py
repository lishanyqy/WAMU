"""DualSystem cross-attention onto the video DiT's own K/V (``variant: cross_attention_direct``).

Same plan as :mod:`joint_cross_attn` (the video DiT runs to completion, a separate ActionDiT runs afterwards with
self-attn -> cross-attn to video -> context cross-attn -> FFN per block, two separate softmaxes) except for WHAT the
action cross-attention reads:

- ``joint_cross_attn``: video block OUTPUT hidden states -> action-side ``video_projs`` (3072 -> bridge_kv_dim) ->
  action-side ``k`` / ``v`` projections (all newly initialised);
- ``cross_attention_direct``: the video DiT's OWN keys/values at each bridge layer, i.e. exactly the K/V that video
  block's self-attention uses (``self_attn.k`` / ``.v`` + key RMSNorm + 3D RoPE, from ``pre_attn_at_layer`` on that
  block's input). No ``video_projs`` and no bridge K/V projections; the action side only adds the query ``q`` /
  ``norm_q`` and the output ``o``.

This isolates "video's own K/V space" (what ``joint_self_attn`` uses) from "joint softmax / interleaved MoT".

Consequences:
- ``num_heads`` / ``attn_head_dim`` of the action DiT must equal the video DiT's (the default; overrides are rejected).
- ``bridge_kv_dim`` and ``bridge_joint_softmax`` do not apply (``bridge_kv_dim`` is ignored, joint softmax rejected).
- The video stream is computed exactly as in ``joint_cross_attn`` (``run_block`` untouched); the K/V are recomputed
  once more from the same block input for the action stream (an extra norm + Q/K/V projection per bridge layer;
  under gradient checkpointing that recompute is checkpointed too).
- The action loss reaches the video DiT through its ``self_attn.k`` / ``.v`` / ``norm_k``, ``norm1`` / modulation and
  the block input, unless ``detach_bridge: true`` (the K/V are then detached).
- ``action_video_attention_mode: first_frame`` keeps the first frame's keys/values; ``bridge_noise`` corrupts a copy of
  the block input and recomputes the K/V from it (as ``joint_self_attn`` does), the video rows stay clean.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture, _cross_attn_options
from openwam.model.architectures.registry import register_architecture

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_cross_attn_direct",
    status="supported",
    note="DualSystem cross-attention whose action query reads the video DiT's own K/V (no bridge projection).",
    framework="dual_system",
    variant="cross_attention_direct",
    options_from_cfg=_cross_attn_options,
)
class DualSystemCrossAttnDirectArchitecture(DualSystemCrossAttnArchitecture):
    _DIRECT_VIDEO_KV = True

    def __init__(self, cfg=None):
        if cfg is not None and bool(cfg.get("bridge_joint_softmax", False)):
            raise ValueError("cross_attention_direct does not support bridge_joint_softmax.")
        super().__init__(cfg)
        if cfg is not None and self.video_backbone is not None:
            self._check_head_geometry()

    def _check_head_geometry(self) -> None:
        vb, ab = self.video_backbone, self.action_backbone
        if ab is None:
            return
        if int(ab.num_heads) != int(vb.num_heads) or int(ab.head_dim) != int(vb.head_dim):
            raise ValueError(
                "cross_attention_direct reads the video DiT's own K/V, so the action DiT must use the same head split: "
                f"action num_heads={ab.num_heads}, head_dim={ab.head_dim} vs video num_heads={vb.num_heads}, "
                f"head_dim={vb.head_dim}. Drop the num_heads / attn_head_dim overrides."
            )

    def _video_kv_at_layer(
        self,
        layer_id: int,
        vstate,
        noise_plan,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> Tuple[Tensor, Tensor]:
        """The (k, v) the video block ``layer_id`` attends with, computed from its input hidden state."""
        vb = self.video_backbone

        def _kv(hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
            st = copy.copy(vstate)                      # shallow: only hidden_states is replaced
            st.hidden_states = hidden_states
            _, k, v, _ = vb.pre_attn_at_layer(layer_id, st)
            return k, v

        hs = vstate.hidden_states
        if noise_plan is not None:
            # corrupt a copy of the block input for the action stream only (same plan as the other dual variants)
            hs = noise_plan.apply(hs, layer_id) if hs.ndim == 3 else noise_plan.apply(hs.flatten(1, -2), layer_id).view_as(hs)

        if self.training and use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    k, v = torch.utils.checkpoint.checkpoint(_kv, hs, use_reentrant=False)
            else:
                k, v = torch.utils.checkpoint.checkpoint(_kv, hs, use_reentrant=False)
        else:
            k, v = _kv(hs)

        if self._action_video_attention_mode == "first_frame":
            tokens_per_frame = int(vstate.grid_height) * int(vstate.grid_width)
            if tokens_per_frame <= 0:
                raise RuntimeError(
                    "Cannot select the first video frame from the video K/V "
                    f"with grid_height={vstate.grid_height}, grid_width={vstate.grid_width}."
                )
            k, v = k[:, :tokens_per_frame], v[:, :tokens_per_frame]
        if self._detach_bridge:
            k, v = k.detach(), v.detach()
        return k, v

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        noise_ids = pipeline_inputs.pop("_bridge_noise_ids", None)
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )
        self._check_head_geometry()

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        bridge_set = frozenset(ab.bridge_layers)
        video_kv: dict[int, Tuple[Tensor, Tensor]] = {}
        noise_plan = None
        if self._bridge_noise is not None and self.training:
            grid = (int(vstate.grid_frames), int(vstate.grid_height), int(vstate.grid_width))
            noise_plan = self._bridge_noise.plan(noisy_actions.shape[0], grid, noise_ids, noisy_actions.device)

        for block_id in range(vb.num_layers):
            if block_id in bridge_set:
                # BEFORE run_block: the K/V of this block are a function of its input hidden state
                video_kv[block_id] = self._video_kv_at_layer(
                    block_id,
                    vstate,
                    noise_plan,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
            vstate = vb.run_block(block_id, vstate)

        video_pred = vb.finalize(vstate)
        if not video_kv:
            return video_pred, None

        action_pred = self._predict_actions_from_bridges(
            noisy_actions,
            video_kv,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred


__all__ = ["DualSystemCrossAttnDirectArchitecture"]
