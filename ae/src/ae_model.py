"""Small recurrent masked actor and AE observation preprocessing.

This file is used by ``ae_manager.py`` at inference time and by the
training notebook when exporting a checkpoint.  Keep it lightweight: the
competition server calls ``AEManager.ae`` once per AE action, so deployment
should be a single small forward pass plus action masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - lets heuristic fallback still import
    torch = None
    nn = None


# Action ids from til_environment.actions.Action. Duplicated here so inference
# does not need til_environment installed inside the AE container.
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5
NUM_ACTIONS = 6

# ViewChannel ids from til_environment.observation.ViewChannel.
VISIBLE = 0
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ENEMY_AGENT = 10
ENEMY_BASE = 12
ALLY_BOMB = 17
ENEMY_BOMB = 18
ALLY_BOMB_TIMER = 19
ENEMY_BOMB_TIMER = 20


@dataclass(frozen=True)
class PreprocessConfig:
    grid_size: float = 16.0
    max_agent_health: float = 60.0
    max_base_health: float = 100.0
    max_freeze_turns: float = 3.0
    max_iters: float = 200.0
    bomb_cost: float = 1.5
    max_bombs_scale: float = 10.0


def _as_np(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < size:
        out[int(index)] = 1.0
    return out


def preprocess_observation(
    observation: dict[str, Any],
    cfg: PreprocessConfig = PreprocessConfig(),
    *,
    device: str | "torch.device" = "cpu",
) -> dict[str, "torch.Tensor"]:
    """Convert the challenge observation dict into batched tensors.

    Returned tensors have batch dimension 1.  The actor accepts dynamic
    viewcone sizes, so this works with both qualifier README shapes and the
    current finals environment config.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for neural inference.")

    agent_view = _as_np(observation["agent_viewcone"])
    base_view = _as_np(observation["base_viewcone"])
    action_mask = _as_np(observation.get("action_mask", np.ones(NUM_ACTIONS)), np.float32)

    location = _as_np(observation.get("location", [0, 0]))
    base_location = _as_np(observation.get("base_location", [0, 0]))
    health = float(_as_np(observation.get("health", [0.0])).reshape(-1)[0])
    base_health = float(_as_np(observation.get("base_health", [0.0])).reshape(-1)[0])
    team_resources = float(_as_np(observation.get("team_resources", [0.0])).reshape(-1)[0])
    frozen_ticks = float(observation.get("frozen_ticks", 0))
    team_bombs = float(observation.get("team_bombs", 0))
    step = float(observation.get("step", 0))

    scalar = np.concatenate(
        [
            _one_hot(int(observation.get("direction", 0)), 4),
            np.clip(location / cfg.grid_size, 0.0, 1.0),
            np.clip(base_location / cfg.grid_size, 0.0, 1.0),
            np.array(
                [
                    np.clip(health / cfg.max_agent_health, 0.0, 1.0),
                    np.clip(frozen_ticks / cfg.max_freeze_turns, 0.0, 1.0),
                    np.clip(base_health / cfg.max_base_health, 0.0, 1.0),
                    np.clip(team_resources / cfg.bomb_cost, 0.0, 10.0) / 10.0,
                    np.clip(team_bombs / cfg.max_bombs_scale, 0.0, 1.0),
                    np.clip(step / cfg.max_iters, 0.0, 1.0),
                ],
                dtype=np.float32,
            ),
            np.clip(action_mask, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "agent_view": torch.from_numpy(agent_view).float().unsqueeze(0).to(device),
        "base_view": torch.from_numpy(base_view).float().unsqueeze(0).to(device),
        "scalar": torch.from_numpy(scalar).float().unsqueeze(0).to(device),
        "action_mask": torch.from_numpy(action_mask).float().unsqueeze(0).to(device),
    }


if nn is not None:

    class ViewEncoder(nn.Module):
        """Tiny CNN that accepts HWC tensors and produces a fixed embedding."""

        def __init__(self, in_channels: int = 25, hidden: int = 64, out_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
                nn.Linear(hidden * 4, out_dim),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Input comes in as NHWC from the challenge observation.
            if x.ndim != 4:
                raise ValueError(f"Expected NHWC tensor, got shape {tuple(x.shape)}")
            x = x.permute(0, 3, 1, 2).contiguous()
            return self.net(x)


    class MaskedRecurrentActor(nn.Module):
        """Recurrent policy used for deployed AE inference.

        Checkpoints exported by the notebook should contain this module's
        ``state_dict`` under the key ``actor_state_dict``.
        """

        def __init__(
            self,
            scalar_dim: int = 20,
            hidden_dim: int = 256,
            num_actions: int = NUM_ACTIONS,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.agent_encoder = ViewEncoder(25, 64, 128)
            self.base_encoder = ViewEncoder(25, 32, 64)
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 96),
                nn.ReLU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(128 + 64 + 96, hidden_dim),
                nn.ReLU(),
            )
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
            self.policy = nn.Linear(hidden_dim, num_actions)

        def initial_hidden(
            self,
            batch_size: int = 1,
            *,
            device: str | "torch.device" | None = None,
        ) -> "torch.Tensor":
            if device is None:
                device = next(self.parameters()).device
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        def forward(
            self,
            batch: dict[str, "torch.Tensor"],
            hidden: "torch.Tensor | None" = None,
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            agent_z = self.agent_encoder(batch["agent_view"])
            base_z = self.base_encoder(batch["base_view"])
            scalar_z = self.scalar_encoder(batch["scalar"])
            fused = self.fuse(torch.cat([agent_z, base_z, scalar_z], dim=-1))

            if hidden is None:
                hidden = self.initial_hidden(fused.shape[0], device=fused.device)
            next_hidden = self.rnn(fused, hidden)
            logits = self.policy(next_hidden)
            return logits, next_hidden


def load_actor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | "torch.device" = "cpu",
) -> "MaskedRecurrentActor":
    """Load a deployed actor checkpoint produced by the notebook."""

    if torch is None:
        raise RuntimeError("PyTorch is required to load the actor checkpoint.")

    actor = MaskedRecurrentActor().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    actor.load_state_dict(state, strict=False)
    actor.eval()
    return actor


def legal_argmax(logits: "torch.Tensor", action_mask: "torch.Tensor") -> int:
    """Return the argmax action after masking illegal actions."""

    masked_logits = logits.clone()
    masked_logits[action_mask <= 0.0] = -1.0e9
    return int(torch.argmax(masked_logits, dim=-1).item())


def _current_view_cell(view: np.ndarray) -> tuple[int, int]:
    # The finals config is behind=2, left=2.  Fall back to geometric centre for
    # shape variants.
    return min(2, view.shape[0] // 2), min(2, view.shape[1] // 2)


def current_cell_bomb_danger(observation: dict[str, Any], danger_timer: float = 2.0) -> bool:
    """Detect an imminent bomb on the agent's current cell from local channels."""

    view = _as_np(observation.get("agent_viewcone", np.zeros((1, 1, 25), dtype=np.float32)))
    if view.ndim != 3 or view.shape[-1] <= ENEMY_BOMB_TIMER:
        return False

    r, c = _current_view_cell(view)
    own_bomb = view[r, c, ALLY_BOMB] > 0.5
    enemy_bomb = view[r, c, ENEMY_BOMB] > 0.5
    if not (own_bomb or enemy_bomb):
        return False

    timers = []
    if own_bomb:
        timers.append(float(view[r, c, ALLY_BOMB_TIMER]))
    if enemy_bomb:
        timers.append(float(view[r, c, ENEMY_BOMB_TIMER]))
    return bool(timers and min(timers) <= danger_timer)


def enemy_close_in_view(observation: dict[str, Any], radius: int = 1) -> bool:
    """Cheap combat feature for fallback mode."""

    view = _as_np(observation.get("agent_viewcone", np.zeros((1, 1, 25), dtype=np.float32)))
    if view.ndim != 3 or view.shape[-1] <= ENEMY_BASE:
        return False
    r, c = _current_view_cell(view)
    r0, r1 = max(0, r - radius), min(view.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(view.shape[1], c + radius + 1)
    patch = view[r0:r1, c0:c1]
    return bool((patch[..., ENEMY_AGENT].max() > 0.5) or (patch[..., ENEMY_BASE].max() > 0.5))


def choose_heuristic_action(observation: dict[str, Any]) -> int:
    """Safe deterministic fallback before a trained checkpoint is available."""

    mask = _as_np(observation.get("action_mask", np.ones(NUM_ACTIONS)), np.float32)
    legal = [a for a in range(NUM_ACTIONS) if a < len(mask) and mask[a] > 0.0]
    if not legal:
        return STAY

    frozen = int(observation.get("frozen_ticks", 0)) > 0
    if frozen and STAY in legal:
        return STAY

    # If standing on a bomb that is close to detonation, leave immediately.
    if current_cell_bomb_danger(observation):
        for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
            if action in legal:
                return action

    # If an enemy/base is adjacent and a bomb is legal, be aggressive.
    if PLACE_BOMB in legal and enemy_close_in_view(observation, radius=1):
        return PLACE_BOMB

    # Score-oriented fallback: keep moving through the maze.
    if FORWARD in legal:
        return FORWARD

    # Alternate turns by step to avoid getting stuck in one spin direction.
    step = int(observation.get("step", 0))
    turn_order = (LEFT, RIGHT) if step % 2 == 0 else (RIGHT, LEFT)
    for action in (*turn_order, BACKWARD, STAY):
        if action in legal:
            return action
    return int(legal[0])


def apply_safety_veto(
    observation: dict[str, Any],
    proposed_action: int,
    ranked_actions: list[int] | None = None,
) -> int:
    """Minimal inference-time wrapper.

    This intentionally avoids being a full planner.  It only vetoes illegal
    actions and obvious "stand/turn on an about-to-explode bomb" cases.
    """

    mask = _as_np(observation.get("action_mask", np.ones(NUM_ACTIONS)), np.float32)
    legal = {a for a in range(NUM_ACTIONS) if a < len(mask) and mask[a] > 0.0}
    if not legal:
        return STAY

    if int(observation.get("frozen_ticks", 0)) > 0:
        return STAY if STAY in legal else int(next(iter(legal)))

    action = int(proposed_action)
    if action not in legal:
        if ranked_actions:
            for candidate in ranked_actions:
                if int(candidate) in legal:
                    action = int(candidate)
                    break
        else:
            action = choose_heuristic_action(observation)

    if current_cell_bomb_danger(observation) and action in {LEFT, RIGHT, STAY, PLACE_BOMB}:
        if ranked_actions:
            for candidate in ranked_actions:
                candidate = int(candidate)
                if candidate in legal and candidate in {FORWARD, BACKWARD}:
                    return candidate
        for candidate in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
            if candidate in legal:
                return candidate

    return action
