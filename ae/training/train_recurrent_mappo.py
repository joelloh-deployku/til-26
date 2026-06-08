"""Recurrent masked PPO/MAPPO-style training scaffold for AE.

Run from the repository root:

    python ae/training/train_recurrent_mappo.py --updates 300 --episodes-per-update 16

The script exports the deployed actor to:

    ae/src/artifacts/ae_actor.pt

This is intentionally a compact baseline.  The next high-impact upgrade is to
add your previous best MAPPO checkpoint as a frozen opponent in
``opponent_action`` and to replace ``ValueNet`` with a centralized critic that
sees full environment state during training only.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import trange

REPO = Path(__file__).resolve().parents[2]
AE_SRC = REPO / "ae" / "src"
AE_ARTIFACTS = AE_SRC / "artifacts"
AE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(AE_SRC))
sys.path.insert(0, str(REPO / "til-26-ae"))

from ae_model import (  # noqa: E402
    MaskedRecurrentActor,
    ViewEncoder,
    choose_heuristic_action,
    preprocess_observation,
)
from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config  # noqa: E402


@dataclass
class TrainConfig:
    total_updates: int = 300
    episodes_per_update: int = 16
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 512
    clip_range: float = 0.15
    lr: float = 3e-4
    entropy_coef_start: float = 0.03
    entropy_coef_end: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Keep random/noisy opponents small.  They are robustness data, not a model
    # of the finals meta.
    p_selfplay_opponent: float = 0.35
    p_heuristic_opponent: float = 0.55
    p_random_noisy_opponent: float = 0.10


class ValueNet(nn.Module):
    """Decentralized value head for compact PPO training.

    For stronger MAPPO, replace this with a centralized critic that sees full
    env state or all agents' observations/actions during training.  Deployment
    still exports only ``MaskedRecurrentActor``.
    """

    def __init__(self, scalar_dim: int = 20):
        super().__init__()
        self.agent_encoder = ViewEncoder(25, 64, 128)
        self.base_encoder = ViewEncoder(25, 32, 64)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64 + 96, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agent_z = self.agent_encoder(batch["agent_view"])
        base_z = self.base_encoder(batch["base_view"])
        scalar_z = self.scalar_encoder(batch["scalar"])
        return self.head(torch.cat([agent_z, base_z, scalar_z], dim=-1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int | None = None) -> Bomberman:
    cfg = default_config()
    cfg.env.render_mode = None
    env = Bomberman(cfg)
    env.reset(seed=seed)
    return env


def masked_dist(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    masked_logits = logits.masked_fill(action_mask <= 0.0, -1.0e9)
    return Categorical(logits=masked_logits)


def random_legal_action(observation: dict) -> int:
    mask = np.asarray(observation["action_mask"], dtype=np.float32)
    legal = np.flatnonzero(mask > 0)
    if len(legal) == 0:
        return 4
    return int(np.random.choice(legal))


def sample_opponent_kind(cfg: TrainConfig) -> str:
    r = random.random()
    if r < cfg.p_selfplay_opponent:
        return "selfplay"
    if r < cfg.p_selfplay_opponent + cfg.p_heuristic_opponent:
        return "heuristic"
    return "random"


def opponent_action(
    observation: dict,
    *,
    actor: MaskedRecurrentActor | None,
    hidden: torch.Tensor | None,
    kind: str,
    device: str,
) -> tuple[int, torch.Tensor | None]:
    """Action generator for non-learner agents."""

    if kind == "random":
        return random_legal_action(observation), hidden

    if kind == "selfplay" and actor is not None:
        with torch.no_grad():
            batch = preprocess_observation(observation, device=device)
            if hidden is None:
                hidden = actor.initial_hidden(1, device=device)
            logits, next_hidden = actor(batch, hidden)
            dist = masked_dist(logits, batch["action_mask"])
            action = torch.argmax(dist.probs, dim=-1)
            return int(action.item()), next_hidden.detach()

    return int(choose_heuristic_action(observation)), hidden


def collect_rollout(
    actor: MaskedRecurrentActor,
    value_net: ValueNet,
    cfg: TrainConfig,
    update_idx: int,
    *,
    seed: int,
    device: str,
) -> tuple[list[dict], list[dict]]:
    """Collect learner transitions from several full six-team episodes."""

    actor.eval()
    value_net.eval()
    transitions: list[dict] = []
    episode_stats: list[dict] = []

    for ep in range(cfg.episodes_per_update):
        env = make_env(seed=seed + update_idx * 10_000 + ep)
        learner_agent = random.choice(env.agents)
        opponent_kinds = {a: sample_opponent_kind(cfg) for a in env.agents if a != learner_agent}
        hidden_by_agent = {a: actor.initial_hidden(1, device=device) for a in env.agents}

        learner_return = 0.0
        learner_freezes = 0
        done = False

        while not done:
            round_records = []
            n_agents_this_round = len(env.agents)

            for _ in range(n_agents_this_round):
                agent_id = env.agent_selection
                obs = env.observe(agent_id)

                if agent_id == learner_agent:
                    batch = preprocess_observation(obs, device=device)
                    h_in = hidden_by_agent[agent_id].detach()
                    with torch.no_grad():
                        logits, h_out = actor(batch, h_in)
                        dist = masked_dist(logits, batch["action_mask"])
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        value = value_net(batch)

                    hidden_by_agent[agent_id] = h_out.detach()
                    round_records.append(
                        {
                            "agent": agent_id,
                            "agent_view": batch["agent_view"].cpu(),
                            "base_view": batch["base_view"].cpu(),
                            "scalar": batch["scalar"].cpu(),
                            "action_mask": batch["action_mask"].cpu(),
                            "hidden_in": h_in.cpu(),
                            "action": int(action.item()),
                            "log_prob": float(log_prob.item()),
                            "value": float(value.item()),
                            "done": 0.0,
                        }
                    )
                    env.step(int(action.item()))
                else:
                    kind = opponent_kinds.get(agent_id, "heuristic")
                    action, h_next = opponent_action(
                        obs,
                        actor=actor,
                        hidden=hidden_by_agent.get(agent_id),
                        kind=kind,
                        device=device,
                    )
                    if h_next is not None:
                        hidden_by_agent[agent_id] = h_next
                    env.step(int(action))

            for rec in round_records:
                reward = float(env.rewards.get(rec["agent"], 0.0))
                rec["reward"] = reward
                rec["done"] = float(
                    env.terminations.get(rec["agent"], False)
                    or env.truncations.get(rec["agent"], False)
                )
                transitions.append(rec)
                learner_return += reward

            try:
                obs_now = env.observe(learner_agent)
                learner_freezes += int(obs_now.get("frozen_ticks", 0) > 0)
            except Exception:
                pass

            done = all(
                env.terminations.get(a, False) or env.truncations.get(a, False)
                for a in env.agents
            )

        episode_stats.append(
            {"return": learner_return, "freezes": learner_freezes, "learner_agent": learner_agent}
        )

    return transitions, episode_stats


def compute_gae(transitions: list[dict], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.array([t["reward"] for t in transitions], dtype=np.float32)
    values = np.array([t["value"] for t in transitions], dtype=np.float32)
    dones = np.array([t["done"] for t in transitions], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0

    for i in reversed(range(len(transitions))):
        nonterminal = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[i] = last_gae
        next_value = values[i]

    returns = advantages + values
    return advantages, returns


def make_training_batch(
    transitions: list[dict],
    advantages: np.ndarray,
    returns: np.ndarray,
    indices: np.ndarray,
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def cat(name: str) -> torch.Tensor:
        return torch.cat([transitions[i][name] for i in indices], dim=0).to(device)

    batch = {
        "agent_view": cat("agent_view"),
        "base_view": cat("base_view"),
        "scalar": cat("scalar"),
        "action_mask": cat("action_mask"),
    }
    hidden_in = torch.cat([transitions[i]["hidden_in"] for i in indices], dim=0).to(device)
    actions = torch.tensor([transitions[i]["action"] for i in indices], dtype=torch.long, device=device)
    old_log_probs = torch.tensor(
        [transitions[i]["log_prob"] for i in indices], dtype=torch.float32, device=device
    )
    adv = torch.tensor(advantages[indices], dtype=torch.float32, device=device)
    ret = torch.tensor(returns[indices], dtype=torch.float32, device=device)
    return batch, hidden_in, actions, old_log_probs, adv, ret


def ppo_update(
    actor: MaskedRecurrentActor,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict],
    cfg: TrainConfig,
    update_idx: int,
    *,
    device: str,
) -> dict[str, float]:
    if len(transitions) == 0:
        return {}

    actor.train()
    value_net.train()
    advantages, returns = compute_gae(transitions, cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(transitions)
    entropy_coef = np.interp(
        update_idx,
        [0, max(1, cfg.total_updates - 1)],
        [cfg.entropy_coef_start, cfg.entropy_coef_end],
    )

    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.ppo_epochs):
        order = np.random.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = order[start : start + cfg.minibatch_size]
            batch, hidden_in, actions, old_log_probs, adv, ret = make_training_batch(
                transitions, advantages, returns, idx, device=device
            )

            logits, _ = actor(batch, hidden_in)
            dist = masked_dist(logits, batch["action_mask"])
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            values = value_net(batch)
            value_loss = 0.5 * (ret - values).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(value_net.parameters()), cfg.max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
            stats["loss"].append(float(loss.item()))
            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["entropy"].append(float(entropy.item()))
            stats["approx_kl"].append(float(approx_kl))

    return {k: float(np.mean(v)) for k, v in stats.items()}


def save_actor(actor: MaskedRecurrentActor, path: Path, *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "metadata": {
                "model": "MaskedRecurrentActor",
                "seed": seed,
                "note": "Exported from ae/training/train_recurrent_mappo.py",
            },
        },
        path,
    )
    print(f"Saved actor checkpoint: {path}")


def evaluate(
    actor: MaskedRecurrentActor,
    *,
    episodes: int,
    opponent_kind: str,
    device: str,
    seed: int,
) -> dict[str, float | list[float]]:
    actor.eval()
    returns: list[float] = []
    for ep in range(episodes):
        env = make_env(seed=seed + 90_000 + ep)
        learner_agent = "agent_0"
        hidden = {a: actor.initial_hidden(1, device=device) for a in env.agents}
        total = 0.0
        done = False

        while not done:
            for _ in range(len(env.agents)):
                agent_id = env.agent_selection
                obs = env.observe(agent_id)

                if agent_id == learner_agent:
                    with torch.no_grad():
                        batch = preprocess_observation(obs, device=device)
                        logits, h_next = actor(batch, hidden[agent_id])
                        hidden[agent_id] = h_next.detach()
                        dist = masked_dist(logits, batch["action_mask"])
                        action = int(torch.argmax(dist.probs, dim=-1).item())
                else:
                    action, h_next = opponent_action(
                        obs,
                        actor=actor,
                        hidden=hidden.get(agent_id),
                        kind=opponent_kind,
                        device=device,
                    )
                    if h_next is not None:
                        hidden[agent_id] = h_next

                env.step(int(action))

            total += float(env.rewards.get(learner_agent, 0.0))
            done = all(
                env.terminations.get(a, False) or env.truncations.get(a, False)
                for a in env.agents
            )

        returns.append(total)

    return {"mean": float(np.mean(returns)), "std": float(np.std(returns)), "raw": returns}


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)

    cfg = TrainConfig(total_updates=args.updates, episodes_per_update=args.episodes_per_update)
    cfg.p_selfplay_opponent = args.p_selfplay
    cfg.p_heuristic_opponent = args.p_heuristic
    cfg.p_random_noisy_opponent = args.p_random

    actor = MaskedRecurrentActor().to(device)
    value_net = ValueNet().to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(value_net.parameters()), lr=cfg.lr)

    best_selection_score = -1.0e9
    for update in trange(cfg.total_updates, desc="updates"):
        transitions, ep_stats = collect_rollout(
            actor, value_net, cfg, update, seed=args.seed, device=device
        )
        train_stats = ppo_update(
            actor, value_net, optimizer, transitions, cfg, update, device=device
        )

        if update % args.eval_every == 0:
            eval_stats = evaluate(
                actor,
                episodes=args.eval_episodes,
                opponent_kind="heuristic",
                device=device,
                seed=args.seed,
            )
            mean_train_return = float(np.mean([s["return"] for s in ep_stats])) if ep_stats else 0.0
            print(
                {
                    "update": update,
                    "rollout_transitions": len(transitions),
                    "train_return": round(mean_train_return, 3),
                    "eval_mean": round(float(eval_stats["mean"]), 3),
                    "eval_std": round(float(eval_stats["std"]), 3),
                    **{k: round(v, 4) for k, v in train_stats.items()},
                }
            )

            selection_score = float(eval_stats["mean"]) - 0.25 * float(eval_stats["std"])
            if selection_score > best_selection_score:
                best_selection_score = selection_score
                save_actor(actor, AE_ARTIFACTS / "ae_actor.pt", seed=args.seed)
                save_actor(actor, AE_ARTIFACTS / f"ae_actor_update_{update:04d}.pt", seed=args.seed)

    print("Best selection score:", best_selection_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--p-selfplay", type=float, default=0.35)
    parser.add_argument("--p-heuristic", type=float, default=0.55)
    parser.add_argument("--p-random", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
