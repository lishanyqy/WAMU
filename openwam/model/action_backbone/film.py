import torch.nn as nn


class ConcatActionFiLM(nn.Module):
    """FiLM conditioner driven by the pre-attention action tokens.

    The condition is the ActionDiT residual stream entering the layer
    (``x_action``, width ``action_dim``), not the mixed-attention output.
    Each action token is projected ``action_dim -> action_dim // token_reduction``;
    all tokens are then flattened (concatenated) and an MLP produces per-channel
    ``gamma`` / ``beta`` for the video attention output (width ``attn_dim``).
    The flatten gives every timestep its own input weights, so the action
    sequence length is baked in and must equal ``num_action_tokens``.
    """

    def __init__(
        self,
        attn_dim: int,
        action_dim: int,
        cond_dim: int,
        num_action_tokens: int,
        token_reduction: int = 8,
    ):
        super().__init__()
        if num_action_tokens <= 0:
            raise ValueError(f"ConcatActionFiLM: num_action_tokens must be > 0, got {num_action_tokens}")
        if token_reduction <= 0 or action_dim % token_reduction != 0:
            raise ValueError(
                f"ConcatActionFiLM: action_dim ({action_dim}) must be divisible by token_reduction ({token_reduction})"
            )

        self.attn_dim = int(attn_dim)
        self.action_dim = int(action_dim)
        self.num_action_tokens = int(num_action_tokens)
        reduced_dim = action_dim // token_reduction

        # The residual stream is un-normalized and its scale drifts across
        # layers; normalize before projecting.
        self.action_norm = nn.LayerNorm(action_dim, elementwise_affine=False)

        # [B, Sa, Da] -> [B, Sa, Da // r]
        self.token_proj = nn.Linear(action_dim, reduced_dim)

        # [B, Sa * Da // r] -> [B, cond_dim]
        self.action_encoder = nn.Sequential(
            nn.Linear(self.num_action_tokens * reduced_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # [B, cond_dim] -> [B, 2 * attn_dim]
        self.film_head = nn.Linear(cond_dim, 2 * attn_dim)

        # 初始状态为 identity modulation
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

    def forward(self, attn_v, x_action):
        """
        attn_v:   [B, Sv, attn_dim]    video mixed-attention output
        x_action: [B, Sa, action_dim]  action tokens entering this layer (pre-attention)
        """
        if attn_v.ndim != 3 or x_action.ndim != 3:
            raise ValueError(
                f"ConcatActionFiLM expects (B, S, D) tensors, got video={tuple(attn_v.shape)} "
                f"and action={tuple(x_action.shape)}"
            )
        if (
            attn_v.shape[0] != x_action.shape[0]
            or attn_v.shape[-1] != self.attn_dim
            or x_action.shape[-1] != self.action_dim
        ):
            raise ValueError(
                "ConcatActionFiLM shape mismatch: expected video (B, Sv, "
                f"{self.attn_dim}) and action (B, Sa, {self.action_dim}); "
                f"got video={tuple(attn_v.shape)}, action={tuple(x_action.shape)}"
            )
        if x_action.shape[1] != self.num_action_tokens:
            raise ValueError(
                f"ConcatActionFiLM expects {self.num_action_tokens} action tokens "
                f"(film_num_action_tokens), got {x_action.shape[1]}"
            )

        # [B, Sa, Da] -> [B, Sa, Da//r] -> [B, Sa * Da//r] -> [B, Dc]
        action_condition = self.token_proj(self.action_norm(x_action)).flatten(1)
        action_condition = self.action_encoder(action_condition)

        gamma_beta = self.film_head(action_condition)              # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                  # [B, D]

        # 在 video token 维度广播
        gamma = gamma.unsqueeze(1)                                # [B, 1, D]
        beta = beta.unsqueeze(1)                                  # [B, 1, D]

        # FiLM
        attn_v = (1.0 + gamma) * attn_v + beta                    # [B, Sv, D]

        return attn_v
