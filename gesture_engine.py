"""Zero-error thumb-and-middle-finger release-edge gesture engine.

Implements a 4-stage verification pipeline immune to camera distance changes,
sensor jitter, hand-closing false positives, and rapid multi-trigger bounce:
1. Scale-normalized Euclidean distance relative to palm length (Wrist 0 -> Middle MCP 9).
2. Fast Exponential Moving Average (EMA) smoothing on pinch ratio (alpha = 0.82)
   while returning 1:1 unsmoothed pixel corners for zero-lag portal tracking.
3. Relaxed anti-fist guard requiring Index Finger Tip (Landmark 8) > 0.85 * L_ref.
4. Release-edge Schmitt trigger (IDLE -> PINCH_HELD when R < 0.29 for 1 frame;
   holds indefinitely with zero repeated triggers; emits triggered=True once on
   separation R > 0.38 after >= 250 ms cooldown; silently cancels to IDLE if the
   index finger folds below 0.78 * L_ref).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np


WRIST_IDX = 0
THUMB_TIP_IDX = 4
INDEX_MCP_IDX = 5
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9
MIDDLE_TIP_IDX = 12
PINKY_MCP_IDX = 17


@dataclass(frozen=True)
class HandTelemetry:
    """Per-frame gesture telemetry and smoothed pixel coordinates for one hand."""

    hand_label: str
    raw_ratio: float
    smoothed_ratio: float
    palm_ref_length: float
    trigger_distance: float
    thumb_tip_px: Tuple[int, int]
    middle_tip_px: Tuple[int, int]
    index_tip_px: Tuple[int, int]
    wrist_px: Tuple[int, int]
    midpoint_px: Tuple[int, int]
    is_index_extended: bool
    state: str
    consecutive_below: int
    triggered_this_frame: bool


@dataclass
class _HandTrackState:
    """Mutable per-hand EMA filter and Schmitt trigger state."""

    ema_thumb: np.ndarray
    ema_middle: np.ndarray
    ema_index: np.ndarray
    ema_ratio: float
    state: str = "IDLE"
    consecutive_frames: int = 0
    last_seen_time: float = 0.0


def _extract_xy_px(landmark: object, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert a normalized MediaPipe landmark or (x, y) tuple to pixel coordinates."""
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        x = float(landmark.x) * frame_w
        y = float(landmark.y) * frame_h
    else:
        x = float(landmark[0]) * frame_w
        y = float(landmark[1]) * frame_h
    return np.array([x, y], dtype=np.float64)


def _get_landmark_list(hand_landmarks: object) -> Sequence[object]:
    """Support both MediaPipe NormalizedLandmarkList (.landmark) and raw sequences."""
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    return hand_landmarks  # type: ignore[return-value]


class ZeroErrorPinchDetector:
    """Scale-invariant, hysteresis-guarded Thumb + Middle Finger tap detector."""

    def __init__(
        self,
        trigger_threshold: float = 0.29,
        release_threshold: float = 0.38,
        ema_alpha: float = 0.82,
        required_consecutive_frames: int = 1,
        cooldown_seconds: float = 0.25,
        index_extension_factor: float = 0.85,
        index_cancel_factor: float = 0.78,
    ) -> None:
        if not 0.0 < trigger_threshold < release_threshold:
            raise ValueError("Thresholds must satisfy 0 < trigger_threshold < release_threshold.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if required_consecutive_frames < 1:
            raise ValueError("required_consecutive_frames must be >= 1.")

        self.trigger_threshold = float(trigger_threshold)
        self.release_threshold = float(release_threshold)
        self.ema_alpha = float(ema_alpha)
        self.required_consecutive_frames = int(required_consecutive_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self.index_extension_factor = float(index_extension_factor)
        self.index_cancel_factor = float(index_cancel_factor)

        self._tracks: Dict[str, _HandTrackState] = {}
        self.last_trigger_time: float = -1e9

    def process_hand(
        self,
        hand_landmarks: object,
        hand_label: str,
        frame_w: int,
        frame_h: int,
        now: Optional[float] = None,
        track_key: Optional[str] = None,
    ) -> HandTelemetry:
        """Evaluate one detected hand for portal corner tracking and pinch triggering."""
        timestamp = time.perf_counter() if now is None else float(now)
        key = track_key or hand_label
        landmarks = _get_landmark_list(hand_landmarks)

        wrist_px = _extract_xy_px(landmarks[WRIST_IDX], frame_w, frame_h)
        thumb_raw_px = _extract_xy_px(landmarks[THUMB_TIP_IDX], frame_w, frame_h)
        index_raw_px = _extract_xy_px(landmarks[INDEX_TIP_IDX], frame_w, frame_h)
        middle_mcp_px = _extract_xy_px(landmarks[MIDDLE_MCP_IDX], frame_w, frame_h)
        middle_raw_px = _extract_xy_px(landmarks[MIDDLE_TIP_IDX], frame_w, frame_h)

        # Orientation-invariant reference palm scale L_ref:
        # combine longitudinal (Wrist 0 -> Middle MCP 9) and transverse (Index MCP 5 -> Pinky MCP 17)
        # so pitching the hand toward the camera does not collapse palm_ref_length.
        longitudinal = float(np.linalg.norm(middle_mcp_px - wrist_px))
        if len(landmarks) > PINKY_MCP_IDX:
            index_mcp_px = _extract_xy_px(landmarks[INDEX_MCP_IDX], frame_w, frame_h)
            pinky_mcp_px = _extract_xy_px(landmarks[PINKY_MCP_IDX], frame_w, frame_h)
            transverse = float(np.linalg.norm(index_mcp_px - pinky_mcp_px)) * 1.15
        else:
            transverse = 0.0
        palm_ref_length = max(longitudinal, transverse, 1e-5)

        # Anti-fist guard: Index Tip (8) must remain extended relative to palm length
        index_dist = float(np.linalg.norm(index_raw_px - wrist_px))
        is_index_extended = index_dist > (self.index_extension_factor * palm_ref_length)

        track = self._tracks.get(key)
        if track is None:
            ema_thumb = thumb_raw_px.copy()
            ema_middle = middle_raw_px.copy()
            ema_index = index_raw_px.copy()
            trigger_dist = float(np.linalg.norm(ema_thumb - ema_middle))
            raw_ratio = trigger_dist / palm_ref_length
            track = _HandTrackState(
                ema_thumb=ema_thumb,
                ema_middle=ema_middle,
                ema_index=ema_index,
                ema_ratio=raw_ratio,
                state="IDLE",
                consecutive_frames=0,
                last_seen_time=timestamp,
            )
            self._tracks[key] = track
        else:
            alpha = self.ema_alpha
            one_minus_alpha = 1.0 - alpha
            track.ema_thumb = alpha * thumb_raw_px + one_minus_alpha * track.ema_thumb
            track.ema_middle = alpha * middle_raw_px + one_minus_alpha * track.ema_middle
            track.ema_index = alpha * index_raw_px + one_minus_alpha * track.ema_index

            trigger_dist = float(np.linalg.norm(track.ema_thumb - track.ema_middle))
            raw_ratio = trigger_dist / palm_ref_length
            track.ema_ratio = alpha * raw_ratio + one_minus_alpha * track.ema_ratio
            track.last_seen_time = timestamp

        triggered_this_frame = False
        ratio = track.ema_ratio

        if track.state == "IDLE":
            if ratio < self.trigger_threshold and is_index_extended:
                track.consecutive_frames += 1
                if track.consecutive_frames >= self.required_consecutive_frames:
                    track.state = "PINCH_HELD"
            else:
                track.consecutive_frames = 0
        else:  # track.state == "PINCH_HELD"
            if index_dist < (self.index_cancel_factor * palm_ref_length):
                # Safety cancel: user folded hand into a fist while pinching
                track.state = "IDLE"
                track.consecutive_frames = 0
            elif ratio > self.release_threshold:
                # Release-edge trigger: switch once upon separating fingers
                track.state = "IDLE"
                track.consecutive_frames = 0
                if (timestamp - self.last_trigger_time) >= self.cooldown_seconds:
                    self.last_trigger_time = timestamp
                    triggered_this_frame = True

        midpoint = 0.5 * (thumb_raw_px + middle_raw_px)

        return HandTelemetry(
            hand_label=hand_label,
            raw_ratio=raw_ratio,
            smoothed_ratio=ratio,
            palm_ref_length=palm_ref_length,
            trigger_distance=trigger_dist,
            thumb_tip_px=(int(round(thumb_raw_px[0])), int(round(thumb_raw_px[1]))),
            middle_tip_px=(int(round(middle_raw_px[0])), int(round(middle_raw_px[1]))),
            index_tip_px=(int(round(index_raw_px[0])), int(round(index_raw_px[1]))),
            wrist_px=(int(round(wrist_px[0])), int(round(wrist_px[1]))),
            midpoint_px=(int(round(midpoint[0])), int(round(midpoint[1]))),
            is_index_extended=is_index_extended,
            state=track.state,
            consecutive_below=track.consecutive_frames,
            triggered_this_frame=triggered_this_frame,
        )

    def reset_missing_hands(self, active_labels: Set[str]) -> None:
        """Clear EMA & trigger state for hands no longer visible in the frame."""
        stale_keys = [key for key in self._tracks if key not in active_labels]
        for key in stale_keys:
            del self._tracks[key]
