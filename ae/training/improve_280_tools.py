"""Helpers for safely improving the AE update-280 champion.

This module is used by ``improve_280_kl_soup.ipynb``.  It provides:

1. Larger checkpoint evaluation.
2. Checkpoint soup / weight averaging.
3. Short KL-anchored fine-tuning from update 280.

The guiding rule: never overwrite ``ae_actor.pt`` unless the candidate beats
raw update 280 on a larger eval.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.auto import trange

REPO = Path(__file__).resolve().parents[2]
AE_SRC = REPO / "ae" / "src"
AE_TRAIN = REPO / "ae" / "training"
AE_ARTIFACTS = AE_SRC / "artifacts"

sys.path.insert(0, str(AE_SRC))
sys.path.insert(0, str(AE_TRAIN))
sys.path.insert(0, str(REPO / "til-26-ae"))

from ae_model import MaskedRecurrentActor, choose_heuristic_action, preprocess_observation  # noqa: E402
import train_recurrent_mappo as base  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPTS = {
    "110": AE_ARTIFACTS / "ae_actor_update_0110.pt",
    "235": AE_ARTIFACTS / "ae_actor_update_0235.pt",
    "280": AE_ARTIFACTS / "ae_actor_update_0280.pt",
    "285": AE_ARTIFACTS / "ae_actor_update_0285.pt",
}


def set_seed(seed: int = 26) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def actor_state_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        return ckpt["actor_state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_actor(path: Path, device: str = DEVICE) -> MaskedRecurrentActor:
    actor = MaskedRecurrentActor().to(device)
    actor.load_state_dict(actor_state_from_checkpoint(path), strict=False)
    actor.eval()
    return actor


def save_actor(actor: MaskedRecurrentActor, path: Path, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": {
                k: v.detach().cpu() for k, v in actor.state_dict().items()
            },
            "metadata": {"model": "MaskedRecurrentActor", "note": note},
        },
        path,
    )
    print(f"Saved: {path}")


def protect(path: Path, name: str) -> Path:
    target = AE_ARTIFACTS / name
    target.write_bytes(path.read_bytes())
    print(f"Copied {path.name} -> {target.name}")
    return target


def risk_adjusted(mean: float, std: float, penalty: float = 0.25) -> float:
    return float(mean - penalty * std)


def evaluate_actor(
    actor: MaskedRecurrentActor,
    *,
    episodes: int = 50,
    opponent_kind: str = "heuristic",
    seed: int = 26,
    device: str = DEVICE,
) -> dict:
    stats = base.evaluate(
        actor,
        episodes=episodes,
        opponent_kind=opponent_kind,
        device=device,
        seed=seed,
    )
    stats["score"] = risk_adjusted(float(stats["mean"]), float(stats["std"]))
    return stats


def evaluate_checkpoint(
    path: Path,
    *,
    episodes: int = 50,
    opponent_kind: str = "heuristic",
    seed: int = 26,
) -> dict:
    actor = load_actor(path)
    stats = evaluate_actor(
        actor, episodes=episodes, opponent_kind=opponent_kind, seed=seed
    )
    print(
        path.name,
        {
            "mean": round(float(stats["mean"]), 3),
            "std": round(float(stats["std"]), 3),
            "score": round(float(stats["score"]), 3),
        },
    )
    return stats


def eval_table(paths: dict[str, Path], episodes: int = 50, seed: int = 26) -> list[dict]:
    rows = []
    for name, path in paths.items():
        if not path.exists():
            print(f"Missing {name}: {path}")
            continue
        stats = evaluate_checkpoint(path, episodes=episodes, seed=seed)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "score": float(stats["score"]),
            }
        )
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    for row in rows:
        print(row)
    return rows


def checkpoint_soup(weighted_paths: list[tuple[float, Path]], out_path: Path) -> Path:
    """Weight-average actor parameters and save a candidate checkpoint."""
    weights = np.array([w for w, _ in weighted_paths], dtype=np.float64)
    weights = weights / weights.sum()

    states = [actor_state_from_checkpoint(path) for _, path in weighted_paths]
    keys = states[0].keys()
    soup_state = {}

    for key in keys:
        tensors = [state[key] for state in states]
        if not torch.is_floating_point(tensors[0]):
            soup_state[key] = tensors[0].clone()
            continue
        acc = torch.zeros_like(tensors[0], dtype=tensors[0].dtype)
        for weight, tensor in zip(weights, tensors):
            acc += float(weight) * tensor
        soup_state[key] = acc

    torch.save(
        {
            "actor_state_dict": soup_state,
            "metadata": {
                "model": "MaskedRecurrentActor",
                "note": "checkpoint soup",
                "members": [(float(w), path.name) for w, path in weighted_paths],
            },
        },
        out_path,
    )
    print(f"Saved soup: {out_path}")
    return out_path


def make_default_soups() -> dict[str, Path]:
    soups = {}
    if CKPTS["280"].exists() and CKPTS["285"].exists():
        soups["soup_080_280_020_285"] = checkpoint_soup(
            [(0.8, CKPTS["280"]), (0.2, CKPTS["285"])],
            AE_ARTIFACTS / "ae_actor_soup_080_280_020_285.pt",
        )
    if CKPTS["280"].exists() and CKPTS["235"].exists():
        soups["soup_080_280_020_235"] = checkpoint_soup(
            [(0.8, CKPTS["280"]), (0.2, CKPTS["235"])],
            AE_ARTIFACTS / "ae_actor_soup_080_280_020_235.pt",
        )
    if CKPTS["280"].exists() and CKPTS["285"].exists() and CKPTS["110"].exists():
        soups["soup_070_280_020_285_010_110"] = checkpoint_soup(
            [(0.7, CKPTS["280"]), (0.2, CKPTS["285"]), (0.1, CKPTS["110"])],
            AE_ARTIFACTS / "ae_actor_soup_070_280_020_285_010_110.pt",
        )
    return soups


@dataclass
class FTConfig:
    updates: int = 40
    episodes_per_update: int = 16
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 3
    minibatch_size: int = 512
    clip_range: float = 0.12
    lr: float = 2e-5
    entropy_coef_start: float = 0.006
    entropy_coef_end: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    kl_anchor_beta: float = 0.03
    p_latest_self: float = 0.25
    p_checkpoint_pool: float = 0.50
    p_heuristic: float = 0.25
    eval_every: int = 2
    eval_episodes: int = 30
    stop_entropy_below: float = 0.55
    stop_mean_below: float = 450.0
    stop_bad_evals: int = 2


def masked_dist_from_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    masked_logits = logits.masked_fill(action_mask <= 0.0, -1.0e9)
    return Categorical(logits=masked_logits)


def policy_kl_anchor(
    new_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    new_logits = new_logits.masked_fill(action_mask <= 0.0, -1.0e9)
    teacher_logits = teacher_logits.masked_fill(action_mask <= 0.0, -1.0e9)
    logp_new = F.log_softmax(new_logits, dim=-1)
    logp_teacher = F.log_softmax(teacher_logits, dim=-1)
    p_teacher = torch.exp(logp_teacher)
    return torch.sum(p_teacher * (logp_teacher - logp_new), dim=-1).mean()


def choose_pool_opponent(checkpoint_pool: list[MaskedRecurrentActor], cfg: FTConfig):
    r = random.random()
    if r < cfg.p_latest_self:
        return "latest", None
    if r < cfg.p_latest_self + cfg.p_checkpoint_pool and checkpoint_pool:
        return "checkpoint", random.choice(checkpoint_pool)
    return "heuristic", None


def opponent_action_ft(
    observation: dict,
    *,
    latest_actor: MaskedRecurrentActor,
    checkpoint_actor: MaskedRecurrentActor | None,
    hidden: torch.Tensor | None,
    kind: str,
    device: str = DEVICE,
) -> tuple[int, torch.Tensor | None]:
    if kind == "heuristic":
        return int(choose_heuristic_action(observation)), hidden

    actor = latest_actor if kind == "latest" else checkpoint_actor
    if actor is None:
        return int(choose_heuristic_action(observation)), hidden

    actor.eval()
    with torch.no_grad():
        batch = preprocess_observation(observation, device=device)
        if hidden is None:
            hidden = actor.initial_hidden(1, device=device)
        logits, next_hidden = actor(batch, hidden)
        dist = masked_dist_from_logits(logits, batch["action_mask"])
        action = int(torch.argmax(dist.probs, dim=-1).item())
    return action, next_hidden.detach()


def collect_rollout_ft(
    actor: MaskedRecurrentActor,
    value_net: base.ValueNet,
    cfg: FTConfig,
    update_idx: int,
    *,
    checkpoint_pool: list[MaskedRecurrentActor],
    seed: int,
    device: str = DEVICE,
):
    actor.eval()
    value_net.eval()
    transitions = []
    episode_stats = []

    for ep in range(cfg.episodes_per_update):
        env = base.make_env(seed=seed + update_idx * 10_000 + ep)
        learner_agent = random.choice(env.agents)
        opponent_specs = {
            a: choose_pool_opponent(checkpoint_pool, cfg)
            for a in env.agents
            if a != learner_agent
        }
        hidden_by_agent = {a: actor.initial_hidden(1, device=device) for a in env.agents}
        learner_return = 0.0
        done = False

        while not done:
            round_records = []
            for _ in range(len(env.agents)):
                agent_id = env.agent_selection
                obs = env.observe(agent_id)

                if agent_id == learner_agent:
                    batch = preprocess_observation(obs, device=device)
                    h_in = hidden_by_agent[agent_id].detach()
                    with torch.no_grad():
                        logits, h_out = actor(batch, h_in)
                        dist = masked_dist_from_logits(logits, batch["action_mask"])
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
                    kind, ckpt_actor = opponent_specs.get(agent_id, ("heuristic", None))
                    action, h_next = opponent_action_ft(
                        obs,
                        latest_actor=actor,
                        checkpoint_actor=ckpt_actor,
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

            done = all(
                env.terminations.get(a, False) or env.truncations.get(a, False)
                for a in env.agents
            )
        episode_stats.append({"return": learner_return, "learner_agent": learner_agent})

    return transitions, episode_stats


def ppo_update_kl(
    actor: MaskedRecurrentActor,
    teacher_actor: MaskedRecurrentActor,
    value_net: base.ValueNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict],
    cfg: FTConfig,
    update_idx: int,
    *,
    device: str = DEVICE,
):
    if not transitions:
        return {}

    actor.train()
    value_net.train()
    teacher_actor.eval()

    advantages, returns = base.compute_gae(transitions, cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(transitions)
    entropy_coef = np.interp(
        update_idx,
        [0, max(1, cfg.updates - 1)],
        [cfg.entropy_coef_start, cfg.entropy_coef_end],
    )
    stats = defaultdict(list)

    for _ in range(cfg.ppo_epochs):
        order = np.random.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = order[start : start + cfg.minibatch_size]
            batch, hidden_in, actions, old_log_probs, adv, ret = base.make_training_batch(
                transitions, advantages, returns, idx, device=device
            )
            logits, _ = actor(batch, hidden_in)
            dist = masked_dist_from_logits(logits, batch["action_mask"])
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            values = value_net(batch)
            value_loss = 0.5 * (ret - values).pow(2).mean()
            with torch.no_grad():
                teacher_logits, _ = teacher_actor(batch, hidden_in)
            kl_anchor = policy_kl_anchor(logits, teacher_logits, batch["action_mask"])
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - entropy_coef * entropy
                + cfg.kl_anchor_beta * kl_anchor
            )
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
            stats["kl_anchor"].append(float(kl_anchor.item()))

    return {k: float(np.mean(v)) for k, v in stats.items()}


def fine_tune_from_280(
    *,
    base_checkpoint: Path = CKPTS["280"],
    checkpoint_pool_paths: list[Path] | None = None,
    cfg: FTConfig = FTConfig(),
    seed: int = 280,
):
    """Short branch fine-tune from update 280.

    Returns ``(best_path, logs)``.  Promote the best path only after a larger
    eval confirms it beats raw 280.
    """
    set_seed(seed)
    actor = load_actor(base_checkpoint, DEVICE)
    teacher_actor = load_actor(base_checkpoint, DEVICE)
    for p in teacher_actor.parameters():
        p.requires_grad_(False)

    value_net = base.ValueNet().to(DEVICE)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value_net.parameters()), lr=cfg.lr
    )

    if checkpoint_pool_paths is None:
        checkpoint_pool_paths = [CKPTS["110"], CKPTS["235"], CKPTS["280"], CKPTS["285"]]
    checkpoint_pool = [load_actor(path, DEVICE) for path in checkpoint_pool_paths if path.exists()]
    for opp in checkpoint_pool:
        opp.eval()
        for p in opp.parameters():
            p.requires_grad_(False)

    baseline = evaluate_actor(actor, episodes=cfg.eval_episodes, seed=seed)
    best_score = float(baseline["score"])
    best_path = AE_ARTIFACTS / "ae_actor_ft_best_from_0280.pt"
    save_actor(actor, best_path, note=f"initial 280 baseline score={best_score:.3f}")
    print("Baseline:", {k: round(float(v), 3) for k, v in baseline.items() if k != "raw"})

    bad_evals = 0
    logs = []
    for update in trange(cfg.updates, desc="ft_updates"):
        transitions, ep_stats = collect_rollout_ft(
            actor,
            value_net,
            cfg,
            update,
            checkpoint_pool=checkpoint_pool,
            seed=seed,
            device=DEVICE,
        )
        train_stats = ppo_update_kl(
            actor, teacher_actor, value_net, optimizer, transitions, cfg, update, device=DEVICE
        )
        if update % cfg.eval_every != 0:
            continue

        eval_stats = evaluate_actor(actor, episodes=cfg.eval_episodes, seed=seed + 1000 + update)
        score = float(eval_stats["score"])
        train_return = float(np.mean([s["return"] for s in ep_stats])) if ep_stats else 0.0
        row = {
            "update": update,
            "train_return": train_return,
            "eval_mean": float(eval_stats["mean"]),
            "eval_std": float(eval_stats["std"]),
            "score": score,
            **train_stats,
        }
        logs.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        if score > best_score:
            best_score = score
            save_actor(actor, best_path, note=f"fine-tuned from 280, score={best_score:.3f}")
            save_actor(
                actor,
                AE_ARTIFACTS / f"ae_actor_ft_from_0280_update_{update:04d}.pt",
                note=f"fine-tuned from 280, score={best_score:.3f}",
            )

        entropy = float(train_stats.get("entropy", 999.0))
        if float(eval_stats["mean"]) < cfg.stop_mean_below:
            bad_evals += 1
        else:
            bad_evals = 0
        if entropy < cfg.stop_entropy_below:
            print(f"Stopping: entropy below threshold: {entropy}")
            break
        if bad_evals >= cfg.stop_bad_evals:
            print("Stopping: repeated weak evals.")
            break

    print("Best fine-tune score:", best_score)
    print("Best fine-tune path:", best_path)
    return best_path, logs
