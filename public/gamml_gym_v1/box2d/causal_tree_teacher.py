"""Small causal-tree teachers for Gymnasium Box2D demos.

These teachers are hand-built demo controllers. They expand a compact primitive
set each step, rank candidates with causal features, and expose the ranking for
video overlays and summaries. They are not trained policies and are not solved
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Candidate:
    name: str
    action: np.ndarray
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    action: np.ndarray
    selected: Candidate
    top_candidates: tuple[Candidate, ...]
    planning_ms: float
    candidate_count: int


class BipedalWalkerCausalTreeTeacher:
    """Think-style primitive scorer for BipedalWalker-v3."""

    env_id = "BipedalWalker-v3"

    def __init__(self) -> None:
        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def primitive_templates(self, obs: np.ndarray) -> list[tuple[str, np.ndarray]]:
        phase = self._phase(obs)
        base = [
            ("coast_balance", [0.0, 0.0, 0.0, 0.0]),
            ("brace_both_knees", [0.0, 0.8, 0.0, 0.8]),
            ("hips_forward_soft", [0.45, 0.25, -0.2, 0.2]),
            ("left_swing_right_stance", [-0.35, 0.85, 0.75, -0.45]),
            ("right_swing_left_stance", [0.75, -0.45, -0.35, 0.85]),
            ("left_push", [0.9, -0.15, -0.15, 0.65]),
            ("right_push", [-0.15, 0.65, 0.9, -0.15]),
            ("recover_lean_back", [-0.75, 0.65, -0.75, 0.65]),
            ("recover_lean_forward", [0.75, 0.25, 0.75, 0.25]),
            ("low_clearance_crouch", [0.15, 1.0, 0.15, 1.0]),
            ("alternating_stride", [0.65, -0.25, -0.55, 0.9] if phase < 0.5 else [-0.55, 0.9, 0.65, -0.25]),
        ]
        return [(name, np.asarray(action, dtype=np.float32)) for name, action in base]

    def decide(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        candidates = [
            self._score_primitive(name, action, obs)
            for name, action in self.primitive_templates(obs)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.step_idx += 1
        return Decision(
            action=candidates[0].action,
            selected=candidates[0],
            top_candidates=tuple(candidates[:4]),
            planning_ms=elapsed,
            candidate_count=len(candidates),
        )

    def _score_primitive(self, name: str, action: np.ndarray, obs: np.ndarray) -> Candidate:
        hull_angle = float(obs[0])
        hull_w = float(obs[1])
        vel_x = float(obs[2])
        vel_y = float(obs[3])
        l_hip = float(obs[4])
        l_knee = float(obs[6])
        l_contact = float(obs[8])
        r_hip = float(obs[9])
        r_knee = float(obs[11])
        r_contact = float(obs[13])
        lidar_min = float(np.min(obs[14:])) if obs.shape[0] >= 24 else 1.0

        reasons: list[str] = []
        score = 0.0

        upright = 1.0 - min(abs(hull_angle) / 0.65, 1.5)
        score += 2.4 * upright
        reasons.append(f"upright={upright:+.2f}")

        forward = np.tanh((vel_x + self._primitive_forward_bias(name, action)) * 1.2)
        score += 2.0 * forward
        reasons.append(f"fwd={forward:+.2f}")

        stability_risk = abs(hull_angle) + 0.35 * abs(hull_w) + 0.12 * max(-vel_y, 0.0)
        score -= 1.7 * stability_risk
        reasons.append(f"risk={stability_risk:.2f}")

        contact_count = l_contact + r_contact
        if contact_count == 0:
            score += 1.6 if "brace" in name or "crouch" in name else -0.9
            reasons.append("airborne_recover")
        elif contact_count == 2:
            score += 0.7 if name in {"left_swing_right_stance", "right_swing_left_stance", "alternating_stride"} else 0.1
            reasons.append("double_contact_stride")
        elif l_contact:
            score += 1.0 if name in {"right_swing_left_stance", "right_push", "alternating_stride"} else -0.2
            reasons.append("left_stance")
        elif r_contact:
            score += 1.0 if name in {"left_swing_right_stance", "left_push", "alternating_stride"} else -0.2
            reasons.append("right_stance")

        if hull_angle > 0.22:
            score += 1.8 if name == "recover_lean_back" else -0.35 * max(action[0] + action[2], 0.0)
            reasons.append("lean_forward")
        elif hull_angle < -0.22:
            score += 1.8 if name == "recover_lean_forward" else -0.35 * max(-(action[0] + action[2]), 0.0)
            reasons.append("lean_back")

        knee_clearance = 0.5 * (abs(l_knee) + abs(r_knee))
        if lidar_min < 0.35:
            score += 0.8 if "crouch" in name or "brace" in name else -0.4
            reasons.append(f"terrain_near={lidar_min:.2f}")
        elif knee_clearance < 0.35 and "stride" in name:
            score += 0.45
            reasons.append("clearance_ok")

        symmetry_penalty = 0.15 * abs(l_hip + r_hip) + 0.08 * float(np.linalg.norm(action))
        score -= symmetry_penalty
        return Candidate(name=name, action=action, score=float(score), reasons=tuple(reasons[:4]))

    def _phase(self, obs: np.ndarray) -> float:
        l_contact = float(obs[8])
        r_contact = float(obs[13])
        if l_contact and not r_contact:
            return 0.75
        if r_contact and not l_contact:
            return 0.25
        return (self.step_idx % 48) / 48.0

    @staticmethod
    def _primitive_forward_bias(name: str, action: np.ndarray) -> float:
        if "stride" in name:
            return 0.35
        if "push" in name:
            return 0.25
        if "swing" in name:
            return 0.18
        return 0.03 * float(action[0] + action[2])


class CarRacingCausalTreeTeacher:
    """Pixel-observation primitive scorer for CarRacing-v3 smoke runs."""

    env_id = "CarRacing-v3"

    def __init__(self) -> None:
        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def primitive_templates(self) -> list[tuple[str, np.ndarray]]:
        primitives = [
            ("gas_straight", [0.0, 0.62, 0.0]),
            ("gas_left_soft", [-0.35, 0.46, 0.0]),
            ("gas_right_soft", [0.35, 0.46, 0.0]),
            ("turn_left_hold", [-0.65, 0.25, 0.0]),
            ("turn_right_hold", [0.65, 0.25, 0.0]),
            ("coast_center", [0.0, 0.0, 0.0]),
            ("brake_stabilize", [0.0, 0.0, 0.22]),
            ("left_brake", [-0.45, 0.0, 0.18]),
            ("right_brake", [0.45, 0.0, 0.18]),
        ]
        return [(name, np.asarray(action, dtype=np.float32)) for name, action in primitives]

    def decide(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        features = self._track_features(obs)
        candidates = [
            self._score_primitive(name, action, features)
            for name, action in self.primitive_templates()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.step_idx += 1
        return Decision(candidates[0].action, candidates[0], tuple(candidates[:4]), elapsed, len(candidates))

    def _score_primitive(self, name: str, action: np.ndarray, f: dict[str, float]) -> Candidate:
        steer = float(action[0])
        gas = float(action[1])
        brake = float(action[2])
        desired_steer = -np.clip(f["track_center_offset"] * 1.4 + f["heading_bias"] * 0.8, -1.0, 1.0)
        score = 0.0
        score += 1.6 * (1.0 - min(abs(steer - desired_steer), 1.4))
        score += 0.9 * gas * f["road_confidence"]
        score -= 0.7 * brake
        offtrack_risk = 1.0 - f["road_confidence"] + abs(f["track_center_offset"])
        if offtrack_risk > 0.65:
            score += 1.0 if brake > 0.1 or abs(steer) > 0.4 else -0.6
        if self.step_idx < 20 and gas > 0.3:
            score += 0.25
        reasons = (
            f"desired_steer={desired_steer:+.2f}",
            f"road_conf={f['road_confidence']:.2f}",
            f"offset={f['track_center_offset']:+.2f}",
            f"risk={offtrack_risk:.2f}",
        )
        return Candidate(name=name, action=action, score=float(score), reasons=reasons)

    @staticmethod
    def _track_features(obs: np.ndarray) -> dict[str, float]:
        img = obs.astype(np.float32) / 255.0
        lower = img[54:92]
        gray = lower.mean(axis=2)
        road = (gray > 0.23) & (gray < 0.72) & (lower[:, :, 1] < 0.82)
        if road.sum() < 20:
            return {"track_center_offset": 0.0, "heading_bias": 0.0, "road_confidence": 0.25}
        ys, xs = np.nonzero(road)
        center = (xs.mean() / max(1, lower.shape[1] - 1)) * 2.0 - 1.0
        near = road[24:]
        far = road[:14]
        near_center = ((np.nonzero(near)[1].mean() / 95.0) * 2.0 - 1.0) if near.sum() else center
        far_center = ((np.nonzero(far)[1].mean() / 95.0) * 2.0 - 1.0) if far.sum() else center
        confidence = min(float(road.mean() * 4.0), 1.0)
        return {
            "track_center_offset": float(center),
            "heading_bias": float(far_center - near_center),
            "road_confidence": confidence,
        }


def format_candidates(candidates: Iterable[Candidate], limit: int = 3) -> list[str]:
    lines = []
    for cand in list(candidates)[:limit]:
        lines.append(f"{cand.name} {cand.score:+.2f}")
    return lines
