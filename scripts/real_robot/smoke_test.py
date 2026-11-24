"""Run one offline π₀ action inference on synthetic observations."""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from openpi.models.model import IMAGE_KEYS, IMAGE_RESOLUTION, Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = WORKSPACE / "models/pi0-lisa-wamu-joint-0p001-v2/30000/model.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Checkpoint file is missing at {args.weights}")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for this smoke test.")

    os.environ.setdefault("OPENPI_DATA_HOME", str(WORKSPACE / ".openpi-cache"))
    device = torch.device("cuda")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(0)
    config = Pi0Config(pytorch_compile_mode=None)
    print(f"Loading {args.weights} on {torch.cuda.get_device_name(device)}", flush=True)
    model = PI0Pytorch(config)
    # This checkpoint omits the expert's unused language-model output projection.
    model.paligemma_with_expert.gemma_expert.lm_head = None
    model.load_state_dict(load_file(args.weights))
    model = model.to(device).eval()

    tokens, token_mask = PaligemmaTokenizer(config.max_token_len).tokenize("Stack the cups.")
    observation = Observation(
        images={key: torch.zeros((1, 3, *IMAGE_RESOLUTION), device=device) for key in IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool, device=device) for key in IMAGE_KEYS},
        state=torch.zeros((1, config.action_dim), device=device),
        tokenized_prompt=torch.as_tensor(tokens, device=device).unsqueeze(0),
        tokenized_prompt_mask=torch.as_tensor(token_mask, device=device).unsqueeze(0),
    )
    with torch.inference_mode():
        actions = model.sample_actions(device, observation)

    expected_shape = (1, config.action_horizon, config.action_dim)
    if tuple(actions.shape) != expected_shape or not torch.isfinite(actions).all().item():
        raise RuntimeError(f"Expected finite actions with shape {expected_shape}, got {tuple(actions.shape)}")
    print(f"Inference passed with {actions.shape[1]} steps × {actions.shape[2]} model action dimensions.")


if __name__ == "__main__":
    main()
