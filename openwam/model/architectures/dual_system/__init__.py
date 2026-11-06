"""DualSystem architecture family.

Three variants split into independent classes:

- :class:`DualSystemCrossAttnArchitecture` — bridge-collection plan, ActionDiT
  runs after the video DiT.
- :class:`DualSystemSelfAttnArchitecture` — interleaved plan, joint ActionDiT
  blocks run inside the video DiT loop.
- :class:`DualSystemIDMArchitecture` — two-stage IDM plan, video denoised first,
  then action denoised with frozen video as condition.
"""

from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
from openwam.model.architectures.dual_system.cross_attention_softmax import DualSystemCrossAttnSoftmaxArchitecture
from openwam.model.architectures.dual_system.cross_attention_direct import DualSystemCrossAttnDirectArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

__all__ = [
    "DualSystemCrossAttnArchitecture",
    "DualSystemCrossAttnSoftmaxArchitecture",
    "DualSystemCrossAttnDirectArchitecture",
    "DualSystemSelfAttnArchitecture",
    "DualSystemIDMArchitecture",
]
