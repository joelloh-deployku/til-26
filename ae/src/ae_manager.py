"""Manages the AE model.

Deployment behaviour:
1. Reset recurrent memory whenever the competition sends an observation with
   ``step == 0``.  The AE server reset endpoint is not called in finals.
2. If ``artifacts/ae_actor.pt`` exists, run the small recurrent masked actor.
3. If no checkpoint is present, use a deterministic safe fallback so the
   container still builds/tests while you are training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ae_model import (
    apply_safety_veto,
    choose_heuristic_action,
    legal_argmax,
    load_actor_checkpoint,
    preprocess_observation,
)

try:
    import torch
except Exception:  # pragma: no cover - fallback mode can still run
    torch = None


class AEManager:
    """Stateful AE inference manager.

    The competition sends exactly one instance per request.  We keep one
    recurrent hidden state and reset it on ``observation["step"] == 0``.
    """

    def __init__(self):
        self.device = "cpu"
        self.checkpoint_path = Path(__file__).resolve().parent / "artifacts" / "ae_actor.pt"
        self.actor = None
        self.hidden = None

        if torch is not None and self.checkpoint_path.exists():
            try:
                # Keep CPU inference for portability on the finals desktop.
                torch.set_num_threads(1)
                self.actor = load_actor_checkpoint(self.checkpoint_path, device=self.device)
                self.hidden = self.actor.initial_hidden(1, device=self.device)
            except Exception as exc:
                # Never fail container startup because of a bad checkpoint.
                print(f"[AEManager] Failed to load {self.checkpoint_path}: {exc}")
                self.actor = None
                self.hidden = None

    def _reset_episode_state(self) -> None:
        if self.actor is not None:
            self.hidden = self.actor.initial_hidden(1, device=self.device)
        else:
            self.hidden = None

    def ae(self, observation: dict[str, int | float | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                ``ae/README.md`` for the format.

        Returns:
            An integer action id:
                0=FORWARD, 1=BACKWARD, 2=LEFT, 3=RIGHT, 4=STAY, 5=PLACE_BOMB.
        """

        if int(observation.get("step", 0)) == 0:
            self._reset_episode_state()

        # Frozen agents may only stay according to the environment action mask.
        if int(observation.get("frozen_ticks", 0)) > 0:
            mask = np.asarray(observation.get("action_mask", [0, 0, 0, 0, 1, 0]))
            if len(mask) > 4 and mask[4] > 0:
                return 4

        if self.actor is None or torch is None:
            return int(choose_heuristic_action(observation))

        try:
            with torch.no_grad():
                batch = preprocess_observation(observation, device=self.device)
                logits, next_hidden = self.actor(batch, self.hidden)
                self.hidden = next_hidden.detach()

                mask = batch["action_mask"]
                action = legal_argmax(logits, mask)

                ranked_actions = (
                    torch.argsort(logits, dim=-1, descending=True)
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )
                return int(apply_safety_veto(observation, action, ranked_actions))
        except Exception as exc:
            # If something unexpected happens mid-match, return a legal action
            # rather than timing out or crashing the container.
            print(f"[AEManager] Inference fallback due to error: {exc}")
            return int(choose_heuristic_action(observation))
