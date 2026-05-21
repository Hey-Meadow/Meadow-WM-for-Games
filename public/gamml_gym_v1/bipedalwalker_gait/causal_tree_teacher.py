"""Phase-conditioned gait macro scorer for BipedalWalker-v3.

This teacher intentionally avoids Box2D snapshot rollout. In this checkout,
``deepcopy(env)`` can be created but stepping the copy raises ``AssertionError``,
so the planner scores gait macros from the 24-D walker observation instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np


SNAPSHOT_ROLLOUT_USED = False
FALLBACK_REASON = "Box2D deepcopy step AssertionError"
CLAIM_LABEL = "causal_tree_teacher_v2_gait_phase"


@dataclass(frozen=True)
class GaitCandidate:
    name: str
    action: np.ndarray
    score: float
    horizon: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GaitDecision:
    action: np.ndarray
    selected: GaitCandidate
    top_candidates: tuple[GaitCandidate, ...]
    planning_ms: float
    candidate_count: int
    phase: str
    phase_value: float
    contact: str
    fall_risk: float
    macro_horizon: int


class BipedalWalkerGaitCausalTreeTeacher:
    """State-based causal scorer over 20-80 frame gait macros.

    The old v1 walker teacher expanded single-step primitives. This version
    scores whole gait phases: stance/swing, push-off, stand, and directional
    recovery. The selected action is the current frame inside the highest
    scoring macro.
    """

    env_id = "BipedalWalker-v3"

    def __init__(self) -> None:
        self.step_idx = 0
        self.phase_value = 0.0
        self.phase_dir = 1.0
        self.last_contact = "air"
        self.last_selected = "stand"

    def reset(self) -> None:
        self.step_idx = 0
        self.phase_value = 0.0
        self.phase_dir = 1.0
        self.last_contact = "air"
        self.last_selected = "stand"

    def decide(self, obs: np.ndarray) -> GaitDecision:
        start = time.perf_counter()
        obs = np.asarray(obs, dtype=np.float32)
        contact = self._contact_label(obs)
        self._advance_phase(obs, contact)
        fall_risk = self._fall_risk(obs)

        candidates = [self._score_macro(name, horizon, obs) for name, horizon in self._macro_specs(obs)]
        candidates.sort(key=lambda c: c.score, reverse=True)
        selected = candidates[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.last_contact = contact
        self.last_selected = selected.name
        self.step_idx += 1
        return GaitDecision(
            action=selected.action,
            selected=selected,
            top_candidates=tuple(candidates[:5]),
            planning_ms=elapsed_ms,
            candidate_count=len(candidates),
            phase=self._phase_label(),
            phase_value=float(self.phase_value),
            contact=contact,
            fall_risk=float(fall_risk),
            macro_horizon=selected.horizon,
        )

    def _macro_specs(self, obs: np.ndarray) -> list[tuple[str, int]]:
        hull_angle = float(obs[0])
        macros = [
            ("stand", 24),
            ("left_stance_right_swing", 64),
            ("right_stance_left_swing", 64),
            ("push_off", 36),
            ("recover_forward", 42),
            ("recover_back", 42),
            ("recover_airborne", 28),
        ]
        if hull_angle > 0.18:
            macros = [("recover_back", 42)] + [m for m in macros if m[0] != "recover_back"]
        elif hull_angle < -0.18:
            macros = [("recover_forward", 42)] + [m for m in macros if m[0] != "recover_forward"]
        return macros

    def _score_macro(self, name: str, horizon: int, obs: np.ndarray) -> GaitCandidate:
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
        contact = self._contact_label(obs)
        phase = self.phase_value

        action = self._macro_action(name, phase, obs)
        energy = float(np.mean(np.square(action)))
        fall_risk = self._fall_risk(obs)

        progress = self._estimate_progress(name, horizon, obs, action)
        angle_ok = 1.0 - min(abs(hull_angle) / 0.75, 1.4)
        angular_ok = 1.0 - min(abs(hull_w) / 1.8, 1.2)
        phase_score = self._phase_match(name, contact, phase)
        contact_score = self._contact_sequence_score(name, l_contact, r_contact)

        score = 0.0
        score += 5.8 * progress
        score += 2.2 * angle_ok
        score += 1.0 * angular_ok
        score += 2.6 * phase_score
        score += 2.0 * contact_score
        score -= 3.6 * fall_risk
        score -= 0.45 * energy
        score -= 0.35 * abs(vel_y)

        if name == "stand":
            score -= 1.2
            score += 1.5 if self.step_idx < 10 or fall_risk > 0.8 else -1.0
            if abs(vel_x) < 0.05 and self.step_idx > 35:
                score -= 1.0
        if "recover" in name:
            score += 2.6 if fall_risk > 0.45 else -0.5
        if name == "recover_back" and hull_angle > 0.12:
            score += 2.0
        if name == "recover_forward" and hull_angle < -0.12:
            score += 2.0
        if name == "recover_airborne" and contact == "air":
            score += 2.5
        if lidar_min < 0.22 and name in {"left_stance_right_swing", "right_stance_left_swing"}:
            score -= 0.6

        knee_lock = max(l_knee, 0.0) + max(r_knee, 0.0)
        if name == "stand" or (action[1] > 0.7 and action[3] > 0.7 and abs(action[0]) < 0.2 and abs(action[2]) < 0.2):
            score -= 1.7 + 0.35 * knee_lock

        reasons = (
            f"prog={progress:+.2f}",
            f"phase={phase_score:+.2f}",
            f"contact={contact_score:+.2f}",
            f"risk={fall_risk:.2f}",
            f"E={energy:.2f}",
        )
        return GaitCandidate(name=name, action=action, score=float(score), horizon=horizon, reasons=reasons)

    def _macro_action(self, name: str, phase: float, obs: np.ndarray) -> np.ndarray:
        s = math.sin(2.0 * math.pi * phase)
        c = math.cos(2.0 * math.pi * phase)
        hull_angle = float(obs[0])
        hull_w = float(obs[1])
        lean_correction = float(np.clip(-1.1 * hull_angle - 0.2 * hull_w, -0.45, 0.45))

        if name == "stand":
            action = [0.0 + lean_correction, 0.55, 0.0 + lean_correction, 0.55]
        elif name == "left_stance_right_swing":
            swing_knee = -0.72 if phase < 0.55 else 0.45
            action = [-0.18 + lean_correction, 0.92, 0.78 + 0.18 * s, swing_knee]
        elif name == "right_stance_left_swing":
            swing_knee = -0.72 if phase < 0.55 else 0.45
            action = [0.78 + 0.18 * s, swing_knee, -0.18 + lean_correction, 0.92]
        elif name == "push_off":
            action = [0.52 + lean_correction, -0.10 + 0.22 * c, -0.48 + lean_correction, 0.76]
            if phase >= 0.5:
                action = [-0.48 + lean_correction, 0.76, 0.52 + lean_correction, -0.10 + 0.22 * c]
        elif name == "recover_forward":
            action = [0.82, 0.30, 0.82, 0.30]
        elif name == "recover_back":
            action = [-0.78, 0.72, -0.78, 0.72]
        elif name == "recover_airborne":
            action = [0.20 + lean_correction, 0.95, -0.20 + lean_correction, 0.95]
        else:
            action = [0.0, 0.0, 0.0, 0.0]
        return np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)

    def _estimate_progress(self, name: str, horizon: int, obs: np.ndarray, action: np.ndarray) -> float:
        vel_x = float(obs[2])
        hull_angle = float(obs[0])
        l_contact = float(obs[8])
        r_contact = float(obs[13])
        contact_count = l_contact + r_contact

        macro_bias = {
            "left_stance_right_swing": 0.23,
            "right_stance_left_swing": 0.23,
            "push_off": 0.27,
            "stand": 0.01,
            "recover_forward": 0.04,
            "recover_back": -0.02,
            "recover_airborne": 0.02,
        }[name]
        stance_gain = 0.12 if contact_count >= 1.0 and name in {"left_stance_right_swing", "right_stance_left_swing", "push_off"} else -0.03
        symmetry_drive = 0.08 * float(action[0] - action[2])
        upright_gate = max(0.0, 1.0 - abs(hull_angle) / 0.75)
        predicted_vx = 0.45 * vel_x + macro_bias + stance_gain + symmetry_drive
        return float(np.clip((predicted_vx * horizon / 64.0) * upright_gate, -0.45, 0.65))

    def _phase_match(self, name: str, contact: str, phase: float) -> float:
        if name == "left_stance_right_swing":
            if contact == "L":
                return 1.0
            if contact == "LR" and phase < 0.5:
                return 0.55
            return -0.25 if contact == "R" else 0.1
        if name == "right_stance_left_swing":
            if contact == "R":
                return 1.0
            if contact == "LR" and phase >= 0.5:
                return 0.55
            return -0.25 if contact == "L" else 0.1
        if name == "push_off":
            return 0.75 if contact == "LR" else 0.25
        if name == "stand":
            return 0.25 if contact == "LR" else -0.2
        if name == "recover_airborne":
            return 0.8 if contact == "air" else -0.15
        return 0.15

    @staticmethod
    def _contact_sequence_score(name: str, l_contact: float, r_contact: float) -> float:
        if name == "left_stance_right_swing":
            return 0.9 if l_contact else (-0.2 if r_contact else 0.1)
        if name == "right_stance_left_swing":
            return 0.9 if r_contact else (-0.2 if l_contact else 0.1)
        if name == "push_off":
            return 0.7 if l_contact + r_contact >= 1.0 else -0.3
        if name == "stand":
            return 0.25 if l_contact + r_contact >= 1.0 else -0.5
        if name == "recover_airborne":
            return 0.8 if l_contact + r_contact == 0.0 else -0.1
        return 0.0

    def _advance_phase(self, obs: np.ndarray, contact: str) -> None:
        if contact == "L" and self.last_contact != "L":
            self.phase_value = 0.12
        elif contact == "R" and self.last_contact != "R":
            self.phase_value = 0.62
        elif contact == "LR":
            self.phase_value = (self.phase_value + 0.018) % 1.0
        elif contact == "air":
            self.phase_value = (self.phase_value + 0.032) % 1.0
        else:
            self.phase_value = (self.phase_value + 0.026) % 1.0

    @staticmethod
    def _contact_label(obs: np.ndarray) -> str:
        l_contact = float(obs[8]) > 0.5
        r_contact = float(obs[13]) > 0.5
        if l_contact and r_contact:
            return "LR"
        if l_contact:
            return "L"
        if r_contact:
            return "R"
        return "air"

    def _phase_label(self) -> str:
        if self.phase_value < 0.25:
            return "left_stance_load"
        if self.phase_value < 0.5:
            return "left_push_right_swing"
        if self.phase_value < 0.75:
            return "right_stance_load"
        return "right_push_left_swing"

    @staticmethod
    def _fall_risk(obs: np.ndarray) -> float:
        hull_angle = float(obs[0])
        hull_w = float(obs[1])
        vel_y = float(obs[3])
        l_contact = float(obs[8])
        r_contact = float(obs[13])
        lidar_min = float(np.min(obs[14:])) if obs.shape[0] >= 24 else 1.0
        risk = 0.0
        risk += min(abs(hull_angle) / 0.9, 1.2) * 0.45
        risk += min(abs(hull_w) / 2.5, 1.0) * 0.20
        risk += min(max(-vel_y, 0.0) / 1.5, 1.0) * 0.15
        risk += 0.15 if l_contact + r_contact == 0.0 else 0.0
        risk += 0.10 if lidar_min < 0.18 else 0.0
        return float(np.clip(risk, 0.0, 1.25))


def format_candidates(candidates: Iterable[GaitCandidate], limit: int = 3) -> list[str]:
    return [f"{cand.name} {cand.score:+.2f}" for cand in list(candidates)[:limit]]
