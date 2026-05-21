"""Track-state causal scorer for CarRacing-v3.

This is a fallback teacher for CarRacing where deepcopy/snapshot rollout is not
usable. It extracts road geometry from the RGB observation, expands a compact
set of action macros, scores them against lane center, curvature, speed, and
offtrack risk, then executes the first action of the selected macro.

The v3 teacher adds a small corner memory.  The road visible in one frame is
not enough for stable racing: if the model sees a left curve for several
frames, it should keep a left-turn commitment briefly instead of treating the
next ambiguous frame as a brand-new problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np


CLAIM_LABEL = "causal_tree_teacher_v3_track_state_corner_memory"
SNAPSHOT_ROLLOUT_USED = False
FALLBACK_REASON = "Box2D deepcopy step AssertionError"
TRACK_FEATURES_USED = [
    "road_mask",
    "centerline_offset",
    "near_road_centroid",
    "mid_road_centroid",
    "far_road_centroid",
    "curvature_proxy",
    "heading_proxy",
    "grass_offtrack_risk",
    "speed_proxy",
    "corner_memory_bias",
    "offset_integral",
    "turn_hold_frames",
]


@dataclass(frozen=True)
class TrackState:
    road_mask: np.ndarray
    centerline_offset: float
    near_road_centroid: float
    mid_road_centroid: float
    far_road_centroid: float
    curvature_proxy: float
    heading_proxy: float
    road_confidence: float
    grass_offtrack_risk: float
    speed_proxy: float
    mask_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Macro:
    name: str
    actions: tuple[tuple[float, float, float], ...]

    @property
    def first_action(self) -> np.ndarray:
        return np.asarray(self.actions[0], dtype=np.float32)


@dataclass(frozen=True)
class Candidate:
    name: str
    first_action: np.ndarray
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    action: np.ndarray
    selected: Candidate
    top_candidates: tuple[Candidate, ...]
    planning_ms: float
    candidate_count: int
    track_state: TrackState
    corner_memory: dict[str, float | int]


def format_candidates(candidates: Iterable[Candidate], limit: int = 4) -> list[str]:
    return [f"{cand.name} {cand.score:+.2f}" for cand in list(candidates)[:limit]]


class CarRacingTrackStateTeacher:
    env_id = "CarRacing-v3"

    def __init__(self) -> None:
        self.step_idx = 0
        self.prev_state: TrackState | None = None
        self.prev_action = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self.corner_bias = 0.0
        self.offset_integral = 0.0
        self.turn_hold_frames = 0
        self.last_turn_sign = 0.0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_state = None
        self.prev_action = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        self.corner_bias = 0.0
        self.offset_integral = 0.0
        self.turn_hold_frames = 0
        self.last_turn_sign = 0.0

    def macros(self) -> list[Macro]:
        return [
            Macro("straight_accel", ((0.00, 0.72, 0.00), (0.00, 0.66, 0.00), (0.00, 0.60, 0.00))),
            Macro("soft_left", ((-0.28, 0.48, 0.00), (-0.24, 0.44, 0.00), (-0.18, 0.42, 0.00))),
            Macro("soft_right", ((0.28, 0.48, 0.00), (0.24, 0.44, 0.00), (0.18, 0.42, 0.00))),
            Macro("hard_left", ((-0.72, 0.26, 0.00), (-0.62, 0.22, 0.00), (-0.48, 0.18, 0.00))),
            Macro("hard_right", ((0.72, 0.26, 0.00), (0.62, 0.22, 0.00), (0.48, 0.18, 0.00))),
            Macro("brake_turn_left", ((-0.56, 0.00, 0.24), (-0.46, 0.00, 0.18), (-0.32, 0.08, 0.08))),
            Macro("brake_turn_right", ((0.56, 0.00, 0.24), (0.46, 0.00, 0.18), (0.32, 0.08, 0.08))),
            Macro("memory_left", ((-0.42, 0.34, 0.00), (-0.38, 0.32, 0.00), (-0.30, 0.36, 0.00))),
            Macro("memory_right", ((0.42, 0.34, 0.00), (0.38, 0.32, 0.00), (0.30, 0.36, 0.00))),
            Macro("tight_memory_left", ((-0.78, 0.08, 0.16), (-0.68, 0.10, 0.10), (-0.50, 0.20, 0.02))),
            Macro("tight_memory_right", ((0.78, 0.08, 0.16), (0.68, 0.10, 0.10), (0.50, 0.20, 0.02))),
            Macro("coast", ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (0.00, 0.10, 0.00))),
        ]

    def decide(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        state = self.extract_track_state(obs, self.prev_state, self.prev_action)
        self._update_memory(state)
        candidates = [self.score_macro(macro, state) for macro in self.macros()]
        candidates.sort(key=lambda cand: cand.score, reverse=True)
        planning_ms = (time.perf_counter() - start) * 1000.0
        memory = self.memory_state()
        decision = Decision(
            action=candidates[0].first_action,
            selected=candidates[0],
            top_candidates=tuple(candidates[:4]),
            planning_ms=float(planning_ms),
            candidate_count=len(candidates),
            track_state=state,
            corner_memory=memory,
        )
        self.prev_state = state
        self.prev_action = decision.action.copy()
        self.step_idx += 1
        if self.turn_hold_frames > 0:
            self.turn_hold_frames -= 1
        return decision

    def score_macro(self, macro: Macro, f: TrackState) -> Candidate:
        actions = np.asarray(macro.actions, dtype=np.float32)
        steer = float(np.mean(actions[:, 0]))
        gas = float(np.mean(actions[:, 1]))
        brake = float(np.mean(actions[:, 2]))
        memory_turn = 0.18 * self.corner_bias + 0.08 * self.offset_integral
        turn_need = f.centerline_offset * 1.05 + f.heading_proxy * 0.90 + f.curvature_proxy * 0.65 + memory_turn
        desired_steer = float(np.clip(turn_need, -1.0, 1.0))
        if self.turn_hold_frames > 0 and abs(self.last_turn_sign) > 0.1:
            desired_steer = float(np.clip(desired_steer + 0.10 * self.last_turn_sign, -1.0, 1.0))
            if abs(desired_steer) < 0.18:
                desired_steer = 0.18 * self.last_turn_sign
        turn_error = abs(steer - desired_steer)
        curve_abs = abs(f.curvature_proxy) + 0.65 * abs(f.heading_proxy)
        risk = float(np.clip(f.grass_offtrack_risk + 0.45 * abs(f.centerline_offset), 0.0, 1.8))
        speed = f.speed_proxy

        score = 0.0
        score += 3.0 * (1.0 - min(turn_error, 1.4) / 1.4)
        score += 1.1 * f.road_confidence
        score -= 1.8 * risk

        if curve_abs < 0.18 and risk < 0.55:
            score += 1.25 * gas - 0.55 * brake
        else:
            score += 0.45 * gas
            score += 1.15 * brake if speed > 0.45 or risk > 0.75 else -0.25 * brake
            score -= 0.65 * max(speed - 0.62, 0.0)

        if risk > 0.85:
            score += 1.25 if brake > 0.08 else -0.85
            score += 0.55 if np.sign(steer) == np.sign(desired_steer) and abs(steer) > 0.35 else -0.25

        macro_sign = -1.0 if "left" in macro.name else (1.0 if "right" in macro.name else 0.0)
        if self.turn_hold_frames > 0 and abs(self.last_turn_sign) > 0.1:
            if macro_sign == self.last_turn_sign:
                score += 0.22 + 0.012 * min(self.turn_hold_frames, 12)
                if "tight_memory" in macro.name and (risk > 0.95 and curve_abs > 0.55):
                    score += 0.35
                if "memory_" in macro.name and risk <= 0.78 and curve_abs > 0.22:
                    score += 0.20
            elif macro_sign != 0.0:
                score -= 0.35

        if abs(self.offset_integral) > 0.30 and macro_sign == np.sign(self.offset_integral):
            score += 0.18
        if speed > 0.58 and (curve_abs > 0.35 or abs(self.corner_bias) > 0.32):
            score += 0.65 if brake > 0.08 else -0.55
        if "tight_memory" in macro.name and not (risk > 0.88 and curve_abs > 0.45):
            score -= 2.2
        if "memory_" in macro.name and abs(desired_steer) < 0.18:
            score -= 0.7

        if self.step_idx < 18 and f.road_confidence > 0.2 and "accel" in macro.name:
            score += 0.35

        reasons = (
            f"desired={desired_steer:+.2f}",
            f"err={turn_error:.2f}",
            f"risk={risk:.2f}",
            f"speed={speed:.2f}",
            f"mem={self.corner_bias:+.2f}/{self.turn_hold_frames}",
        )
        return Candidate(macro.name, macro.first_action, float(score), reasons)

    def _update_memory(self, f: TrackState) -> None:
        turn_signal = 0.45 * f.heading_proxy + 0.35 * f.curvature_proxy + 0.20 * f.centerline_offset
        if f.road_confidence < 0.14:
            turn_signal = 0.70 * self.corner_bias
        self.corner_bias = float(np.clip(0.90 * self.corner_bias + 0.10 * turn_signal, -1.0, 1.0))
        self.offset_integral = float(np.clip(0.94 * self.offset_integral + 0.06 * f.centerline_offset, -0.85, 0.85))
        strong_turn = abs(turn_signal) > 0.26 or abs(self.corner_bias) > 0.34 or f.grass_offtrack_risk > 0.82
        if strong_turn:
            sign = float(np.sign(turn_signal if abs(turn_signal) > 0.08 else self.corner_bias))
            if sign != 0.0:
                self.last_turn_sign = sign
                extra = int(4 + 8 * min(abs(self.corner_bias) + 0.6 * f.grass_offtrack_risk, 1.0))
                self.turn_hold_frames = max(self.turn_hold_frames, extra)
        elif self.turn_hold_frames <= 0 and abs(self.corner_bias) < 0.07:
            self.last_turn_sign = 0.0

    def memory_state(self) -> dict[str, float | int]:
        return {
            "corner_bias": float(self.corner_bias),
            "offset_integral": float(self.offset_integral),
            "turn_hold_frames": int(self.turn_hold_frames),
            "last_turn_sign": float(self.last_turn_sign),
        }

    @staticmethod
    def extract_track_state(
        obs: np.ndarray,
        prev_state: TrackState | None = None,
        prev_action: np.ndarray | None = None,
    ) -> TrackState:
        img = np.asarray(obs)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        y0, y1 = int(h * 0.24), int(h * 0.88)
        roi = img[y0:y1].astype(np.float32)
        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        gray = (r + g + b) / 3.0
        channel_spread = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

        gray_road = (gray > 42.0) & (gray < 205.0) & (channel_spread < 35.0)
        dark_road = (gray > 18.0) & (gray < 92.0) & (channel_spread < 48.0)
        road = gray_road | dark_road

        yy, xx = np.indices(road.shape)
        center_x = (w - 1) * 0.5
        half_width = np.maximum(12.0, (yy / max(1, road.shape[0] - 1)) * w * 0.58 + w * 0.10)
        visible_trapezoid = np.abs(xx - center_x) <= half_width
        road &= visible_trapezoid

        # Suppress the car nose and dashboard area at the bottom center.
        car_y = yy > road.shape[0] * 0.68
        car_x = np.abs(xx - center_x) < w * 0.16
        road &= ~(car_y & car_x & (gray < 80.0))

        road = _majority_smooth(road)
        near = road[int(road.shape[0] * 0.62) :]
        mid = road[int(road.shape[0] * 0.38) : int(road.shape[0] * 0.68)]
        far = road[: int(road.shape[0] * 0.36)]

        fallback = 0.0 if prev_state is None else prev_state.centerline_offset
        near_c = _centroid_x(near, fallback)
        mid_c = _centroid_x(mid, near_c)
        far_c = _centroid_x(far, mid_c)
        centerline = 0.50 * near_c + 0.32 * mid_c + 0.18 * far_c
        heading = far_c - near_c
        curvature = far_c - 2.0 * mid_c + near_c
        road_conf = float(np.clip(road.mean() * 5.2, 0.0, 1.0))

        lower = road[int(road.shape[0] * 0.56) :]
        lower_conf = float(lower.mean()) if lower.size else 0.0
        center_band = road[:, int(w * 0.39) : int(w * 0.61)]
        center_conf = float(center_band.mean()) if center_band.size else 0.0
        green = (g > 90.0) & (g > r * 1.12) & (g > b * 1.12)
        grass_ratio = float(green[visible_trapezoid].mean()) if np.any(visible_trapezoid) else 0.0
        offtrack = float(np.clip((0.42 - lower_conf) * 1.6 + (0.22 - center_conf) * 1.2 + grass_ratio * 0.8, 0.0, 1.0))

        if prev_state is None:
            gas_hint = float(prev_action[1]) if prev_action is not None else 0.0
            speed = 0.18 + 0.25 * gas_hint
        else:
            drift = abs(centerline - prev_state.centerline_offset) + 0.5 * abs(near_c - prev_state.near_road_centroid)
            gas_hint = float(prev_action[1]) if prev_action is not None else 0.0
            brake_hint = float(prev_action[2]) if prev_action is not None else 0.0
            speed = 0.58 * prev_state.speed_proxy + 0.25 * np.clip(drift * 4.0, 0.0, 1.0) + 0.24 * gas_hint - 0.35 * brake_hint
        speed = float(np.clip(speed, 0.0, 1.0))

        return TrackState(
            road_mask=road,
            centerline_offset=float(np.clip(centerline, -1.0, 1.0)),
            near_road_centroid=float(np.clip(near_c, -1.0, 1.0)),
            mid_road_centroid=float(np.clip(mid_c, -1.0, 1.0)),
            far_road_centroid=float(np.clip(far_c, -1.0, 1.0)),
            curvature_proxy=float(np.clip(curvature, -1.0, 1.0)),
            heading_proxy=float(np.clip(heading, -1.0, 1.0)),
            road_confidence=road_conf,
            grass_offtrack_risk=offtrack,
            speed_proxy=speed,
            mask_bbox=(0, y0, w, y1),
        )


def _centroid_x(mask: np.ndarray, fallback: float) -> float:
    if mask.size == 0 or int(mask.sum()) < 6:
        return float(fallback)
    xs = np.nonzero(mask)[1]
    width = max(mask.shape[1] - 1, 1)
    return float((xs.mean() / width) * 2.0 - 1.0)


def _majority_smooth(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="edge")
    total = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return total >= 4
