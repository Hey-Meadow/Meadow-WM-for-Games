"""Causal-tree Think teacher for Gymnasium LunarLander.

The teacher is a transparent state-based controller. Each step expands a small
set of named landing primitives, scores their short projected futures, and
executes the first action of the highest scoring candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np


CLAIM_LABEL = "lunarlander_causal_tree_think_teacher_v1"
ACTION_NAMES = {
    0: "coast",
    1: "left_attitude_engine",
    2: "main_engine",
    3: "right_attitude_engine",
}
PRIMITIVE_NAMES = (
    "main_engine/descent",
    "left_attitude_correction",
    "right_attitude_correction",
    "coast",
    "brake",
    "stabilize/contact",
)


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    chain: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class Candidate:
    name: str
    action: int
    chain: tuple[int, ...]
    score: float
    reasons: dict[str, float]
    projected_states: tuple[tuple[float, float, float, float, float, float], ...]
    predicted_landing_x: float
    predicted_landing_vx: float
    predicted_landing_vy: float
    predicted_landing_steps: int


@dataclass(frozen=True)
class Decision:
    action: int
    selected: Candidate
    top_candidates: tuple[Candidate, ...]
    all_candidates: tuple[Candidate, ...]
    planning_ms: float
    candidate_count: int
    features: dict[str, float]


class LunarLanderCausalTreeTeacher:
    """Explicit primitive scorer for LunarLander-v3/v2 discrete control."""

    def __init__(self, horizon: int = 36) -> None:
        self.horizon = int(horizon)
        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def primitive_templates(self, obs: np.ndarray, features: dict[str, float]) -> list[PrimitiveSpec]:
        del obs
        h = self.horizon
        contact_action = 0
        if features["contact"] > 0.5 and features["vy"] < -0.08:
            contact_action = 2
        elif features["angle_todo"] > 0.035:
            contact_action = 1
        elif features["angle_todo"] < -0.035:
            contact_action = 3

        main_descent = tuple((2 if i in {0, 2, 5, 9, 14, 20, 27} else 0) for i in range(h))
        left_correct = tuple((1 if i < 9 and i % 2 == 0 else 0) for i in range(h))
        right_correct = tuple((3 if i < 9 and i % 2 == 0 else 0) for i in range(h))
        brake = tuple((2 if i < 8 or i in {10, 13, 17, 22, 28} else 0) for i in range(h))
        stabilize = tuple((contact_action if i < 10 and i % 2 == 0 else 0) for i in range(h))
        coast = (0,) * h
        return [
            PrimitiveSpec("main_engine/descent", main_descent, "slow descent while keeping lateral target"),
            PrimitiveSpec("left_attitude_correction", left_correct, "rotate/translate left attitude toward target"),
            PrimitiveSpec("right_attitude_correction", right_correct, "rotate/translate right attitude toward target"),
            PrimitiveSpec("coast", coast, "save fuel when hover and angle errors are small"),
            PrimitiveSpec("brake", brake, "aggressive vertical speed reduction near pad"),
            PrimitiveSpec("stabilize/contact", stabilize, "contact guard after one or both legs touch"),
        ]

    def decide(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        obs = np.asarray(obs, dtype=np.float64)
        features = self.control_features(obs)
        candidates = [
            self.score_primitive(spec, obs, features)
            for spec in self.primitive_templates(obs, features)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.step_idx += 1
        return Decision(
            action=int(candidates[0].action),
            selected=candidates[0],
            top_candidates=tuple(candidates[:5]),
            all_candidates=tuple(candidates),
            planning_ms=float(elapsed_ms),
            candidate_count=len(candidates),
            features=features,
        )

    def control_features(self, obs: np.ndarray) -> dict[str, float]:
        x, y, vx, vy, angle, angular_v, left_contact, right_contact = [float(v) for v in obs[:8]]
        contact = float(left_contact > 0.5 or right_contact > 0.5)

        angle_targ = float(np.clip(0.5 * x + vx, -0.4, 0.4))
        hover_targ = float(0.55 * abs(x))
        angle_todo = (angle_targ - angle) * 0.5 - angular_v
        hover_todo = (hover_targ - y) * 0.5 - 0.5 * vy
        if contact:
            angle_todo = 0.0
            hover_todo = -0.5 * vy

        ideal_action = 0
        if hover_todo > abs(angle_todo) and hover_todo > 0.05:
            ideal_action = 2
        elif angle_todo < -0.05:
            ideal_action = 3
        elif angle_todo > 0.05:
            ideal_action = 1

        descent_risk = max(0.0, -vy - (0.18 + 0.18 * max(y, 0.0)))
        horizontal_risk = abs(x) + 0.8 * abs(vx)
        tilt_risk = abs(angle) + 0.5 * abs(angular_v)
        near_ground = float(y < 0.30)
        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "angle": angle,
            "angular_v": angular_v,
            "left_contact": float(left_contact),
            "right_contact": float(right_contact),
            "contact": contact,
            "angle_targ": angle_targ,
            "hover_targ": hover_targ,
            "angle_todo": float(angle_todo),
            "hover_todo": float(hover_todo),
            "ideal_action": float(ideal_action),
            "descent_risk": float(descent_risk),
            "horizontal_risk": float(horizontal_risk),
            "tilt_risk": float(tilt_risk),
            "near_ground": near_ground,
        }

    def score_primitive(self, spec: PrimitiveSpec, obs: np.ndarray, features: dict[str, float]) -> Candidate:
        projected = self.project(obs, spec.chain, steps=min(self.horizon, 42))
        landing_x, landing_vx, landing_vy, landing_steps = self.estimate_landing(obs, spec.chain)
        final = projected[-1] if projected else tuple(float(v) for v in obs[:6])
        action = int(spec.chain[0])
        ideal = int(features["ideal_action"])

        score = 0.0
        score += 10.0 if action == ideal else -4.0

        hover_todo = features["hover_todo"]
        angle_todo = features["angle_todo"]
        y = features["y"]
        vy = features["vy"]
        contact = features["contact"]
        descent_risk = features["descent_risk"]
        tilt_risk = features["tilt_risk"]

        if spec.name == "main_engine/descent":
            score += 4.5 * max(0.0, hover_todo)
            score += 0.7 * max(0.0, y - 0.25)
            score -= 2.0 * contact
        elif spec.name == "brake":
            score += 7.5 * descent_risk
            score += 2.0 if y < 0.38 and vy < -0.16 else -0.5
            score += 1.6 if y < 0.18 else 0.0
        elif spec.name == "left_attitude_correction":
            score += 7.0 * max(0.0, angle_todo)
            score += 0.7 * tilt_risk
            score -= 2.0 * max(0.0, hover_todo - abs(angle_todo))
        elif spec.name == "right_attitude_correction":
            score += 7.0 * max(0.0, -angle_todo)
            score += 0.7 * tilt_risk
            score -= 2.0 * max(0.0, hover_todo - abs(angle_todo))
        elif spec.name == "coast":
            score += 3.6 if abs(angle_todo) < 0.05 and hover_todo < 0.05 else -1.2
            score += 0.8 if y > 0.20 and descent_risk < 0.10 else -1.4 * descent_risk
            score -= 1.0 if contact and vy < -0.08 else 0.0
        elif spec.name == "stabilize/contact":
            score += 5.2 * contact
            score += 1.8 if y < 0.16 and abs(features["x"]) < 0.25 else -0.3
            score += 1.0 if abs(vy) < 0.18 else -1.2 * max(0.0, -vy - 0.22)

        final_x, _final_y, final_vx, final_vy, final_angle, final_w = final
        landing_penalty = (
            1.8 * abs(landing_x)
            + 1.2 * max(0.0, abs(landing_vx) - 0.28)
            + 1.6 * max(0.0, -landing_vy - 0.38)
            + 0.8 * abs(final_angle)
            + 0.25 * abs(final_w)
        )
        final_speed_penalty = 0.45 * max(0.0, abs(final_vy) - 0.45) + 0.25 * abs(final_vx)
        fuel_cost = sum(0.030 if a in (1, 3) else (0.18 if a == 2 else 0.0) for a in spec.chain[:12])
        score -= landing_penalty + final_speed_penalty + fuel_cost

        reasons = {
            "ideal_action_match": float(action == ideal),
            "hover_todo": float(hover_todo),
            "angle_todo": float(angle_todo),
            "descent_risk": float(descent_risk),
            "landing_x": float(landing_x),
            "landing_vy": float(landing_vy),
            "landing_penalty": float(landing_penalty),
            "fuel_cost": float(fuel_cost),
        }
        return Candidate(
            name=spec.name,
            action=action,
            chain=spec.chain,
            score=float(score),
            reasons=reasons,
            projected_states=tuple(projected),
            predicted_landing_x=float(landing_x),
            predicted_landing_vx=float(landing_vx),
            predicted_landing_vy=float(landing_vy),
            predicted_landing_steps=int(landing_steps),
        )

    def project(
        self,
        obs: np.ndarray,
        chain: tuple[int, ...],
        *,
        steps: int,
        stop_at_ground: bool = False,
    ) -> list[tuple[float, float, float, float, float, float]]:
        x, y, vx, vy, angle, angular_v = [float(v) for v in obs[:6]]
        out: list[tuple[float, float, float, float, float, float]] = []
        for i in range(steps):
            action = int(chain[i]) if i < len(chain) else 0
            if action == 2:
                vy += 0.038 * max(0.12, np.cos(angle))
                vx -= 0.010 * np.sin(angle)
            elif action == 1:
                angular_v += 0.055
                vx -= 0.006
            elif action == 3:
                angular_v -= 0.055
                vx += 0.006

            vy -= 0.010
            vx *= 0.996
            vy *= 0.999
            angular_v *= 0.92
            angle += 0.050 * angular_v
            angle *= 0.998
            x += vx * 0.010
            y += vy * 0.0225
            if y <= 0.0:
                y = 0.0
                vx *= 0.45
                vy *= -0.08
                angular_v *= 0.30
                out.append((x, y, vx, vy, angle, angular_v))
                if stop_at_ground:
                    break
            else:
                out.append((x, y, vx, vy, angle, angular_v))
        return out

    def estimate_landing(self, obs: np.ndarray, chain: tuple[int, ...]) -> tuple[float, float, float, int]:
        extended = tuple(chain) + (0,) * 160
        projected = self.project(obs, extended, steps=min(len(extended), 180), stop_at_ground=True)
        if not projected:
            return float(obs[0]), float(obs[2]), float(obs[3]), 0
        for idx, state in enumerate(projected):
            if state[1] <= 0.0:
                return float(state[0]), float(state[2]), float(state[3]), idx + 1
        last = projected[-1]
        return float(last[0]), float(last[2]), float(last[3]), len(projected)


def format_chain(chain: Iterable[int], limit: int = 12) -> str:
    names = [ACTION_NAMES[int(a)] for a in list(chain)[:limit]]
    compact: list[str] = []
    last = None
    count = 0
    for name in names:
        if name == last:
            count += 1
        else:
            if last is not None:
                compact.append(f"{last}x{count}" if count > 1 else last)
            last = name
            count = 1
    if last is not None:
        compact.append(f"{last}x{count}" if count > 1 else last)
    return " -> ".join(compact)


def format_candidates(candidates: Iterable[Candidate], limit: int = 4) -> str:
    return " | ".join(f"{c.name}:{c.score:+.2f}" for c in list(candidates)[:limit])
