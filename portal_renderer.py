"""High-performance Portal Geometry Engine & Marvel Comic-Book Rendering Engine.

Decouples three core pipelines through lightweight immutable data structures:
1. PortalGeometryEngine: Handedness-aware extraction of ONLY Thumb Tip (4) and
   Index Tip (8) from Left & Right hands, twist-free clockwise [TL, TR, BR, BL]
   ordering, self-intersection repair, degenerate/zero-area rejection, and natural
   unclamped screen-boundary handling.
2. MarvelPortalRenderer: Precomputed uint8 hero Color-LUT cel shading, cached comic
   Ben-Day halftone dot texture, adaptive ink outlines, optional ROI-offset comic face
   filters, and zero-full-frame-copy ROI compositing (< 4 ms per frame).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from face_filter_renderer import MarvelFaceFilterRenderer
from gesture_engine import HandTelemetry


CHARCOAL_BG = (18, 18, 18)        # #121212
CHARCOAL_PANEL = (26, 26, 30)
NEON_ORANGE = (0, 122, 255)       # #FF7A00
NEON_GREEN = (102, 255, 0)        # #00FF66
TEXT_WHITE = (242, 245, 248)
TEXT_MUTED = (160, 168, 178)
CYAN_GLOW = (255, 235, 0)         # #00EBFF in BGR


@dataclass(frozen=True)
class MarvelCharacterProfile:
    """Metadata and comic palette for a Marvel comic dimension."""

    slug: str
    name: str
    accent_bgr: Tuple[int, int, int]
    secondary_bgr: Tuple[int, int, int]
    bgr_tint: Tuple[float, float, float]


@dataclass(frozen=True)
class HandPortalPoints:
    """Lightweight 4-landmark input carrying ONLY Thumb Tip (4) and Index Tip (8)."""

    hand_label: str
    thumb_tip_px: Tuple[int, int]
    index_tip_px: Tuple[int, int]


@dataclass(frozen=True)
class PortalQuadGeometry:
    """Validated 4-point portal quadrilateral and frame-clamped ROI bounds."""

    is_valid: bool
    p1_left_thumb: Optional[Tuple[int, int]]
    p2_left_index: Optional[Tuple[int, int]]
    p3_right_index: Optional[Tuple[int, int]]
    p4_right_thumb: Optional[Tuple[int, int]]
    quad: Optional[np.ndarray]          # (4, 2) int32 clockwise [TL, TR, BR, BL]
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) frame-clamped ROI
    area: float
    rejection_reason: str = ""


@dataclass
class PipelineTimings:
    """Stage-by-stage timing metrics (milliseconds) across the frame pipeline."""

    capture_ms: float = 0.0
    color_convert_ms: float = 0.0
    hand_detect_ms: float = 0.0
    face_detect_ms: float = 0.0
    pose_detect_ms: float = 0.0
    segment_ms: float = 0.0
    portal_geom_ms: float = 0.0
    portal_render_ms: float = 0.0
    comic_shader_ms: float = 0.0
    face_filter_ms: float = 0.0
    record_ms: float = 0.0
    total_frame_ms: float = 0.0
    rendered_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dest_wh: Tuple[int, int] = (0, 0)


@dataclass
class _ShockwaveEffect:
    """Expanding neon shockwave triggered at the thumb-middle pinch midpoint."""

    center: Tuple[int, int]
    accent_bgr: Tuple[int, int, int]
    frame_index: int = 0
    total_frames: int = 6


def _default_profiles() -> List[MarvelCharacterProfile]:
    """Return the 12 Marvel Comic Dimensions in release-snap cycle order."""
    return [
        MarvelCharacterProfile(
            slug="spider_man",
            name="SPIDER-MAN",
            accent_bgr=(40, 35, 245),
            secondary_bgr=(255, 210, 30),
            bgr_tint=(1.15, 0.86, 1.25),
        ),
        MarvelCharacterProfile(
            slug="scarlet_witch",
            name="SCARLET WITCH",
            accent_bgr=(85, 30, 255),
            secondary_bgr=(215, 85, 255),
            bgr_tint=(1.12, 0.72, 1.32),
        ),
        MarvelCharacterProfile(
            slug="doctor_strange",
            name="DOCTOR STRANGE",
            accent_bgr=(0, 145, 255),
            secondary_bgr=(50, 215, 255),
            bgr_tint=(0.88, 1.15, 1.26),
        ),
        MarvelCharacterProfile(
            slug="iron_man",
            name="IRON MAN",
            accent_bgr=(45, 200, 255),
            secondary_bgr=(35, 40, 240),
            bgr_tint=(0.78, 1.16, 1.30),
        ),
        MarvelCharacterProfile("thor", "THOR", (255, 195, 65), (255, 245, 200), (1.28, 1.10, 0.91)),
        MarvelCharacterProfile("loki", "LOKI", (65, 220, 90), (50, 205, 220), (0.78, 1.20, 0.83)),
        MarvelCharacterProfile("black_panther", "BLACK PANTHER", (210, 85, 190), (220, 145, 250), (1.10, 0.78, 1.10)),
        MarvelCharacterProfile("deadpool", "DEADPOOL", (40, 45, 235), (85, 90, 110), (0.84, 0.82, 1.25)),
        MarvelCharacterProfile("captain_america", "CAPTAIN AMERICA", (245, 115, 55), (245, 245, 245), (1.26, 0.90, 0.92)),
        MarvelCharacterProfile("wolverine", "WOLVERINE", (35, 210, 250), (230, 95, 40), (0.82, 1.22, 1.24)),
        MarvelCharacterProfile("venom", "VENOM", (220, 220, 220), (120, 110, 115), (0.76, 0.78, 0.83)),
        MarvelCharacterProfile("ghost_rider", "GHOST RIDER", (0, 130, 255), (0, 235, 255), (0.66, 1.04, 1.35)),
    ]


def _build_hero_lut(bgr_tint: Tuple[float, float, float]) -> np.ndarray:
    """Precompute a 256-entry uint8 3-channel comic cel-shading + hero-tint LUT."""
    vals = np.arange(256, dtype=np.float32)
    # High-contrast comic S-curve + 8-level cel quantization
    norm = vals / 255.0
    contrasted = (0.5 - 0.5 * np.cos(np.pi * np.clip(norm * 1.12 - 0.06, 0.0, 1.0))) * 255.0
    quantized = ((contrasted // 32.0) * 32.0 + 16.0)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, 0, ch] = np.clip(np.round(quantized * bgr_tint[ch]), 0, 255).astype(np.uint8)
    return lut


def _build_halftone_matrix(max_h: int = 1080, max_w: int = 1920, cell: int = 6) -> np.ndarray:
    """Precompute a 3-channel Ben-Day halftone dot darkening mask."""
    tile = np.zeros((cell, cell), dtype=np.uint8)
    cv2.circle(tile, (cell // 2, cell // 2), max(1, cell // 4), 22, -1)
    reps_y = (max_h + cell - 1) // cell
    reps_x = (max_w + cell - 1) // cell
    tiled = np.tile(tile, (reps_y, reps_x))[:max_h, :max_w]
    return np.repeat(tiled[:, :, np.newaxis], 3, axis=2)


def _segments_properly_intersect(
    a: Tuple[int, int],
    b: Tuple[int, int],
    c: Tuple[int, int],
    d: Tuple[int, int],
) -> bool:
    """Return True if closed segments AB and CD properly cross each other."""
    def _orient(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
        val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 1 if val > 0 else (-1 if val < 0 else 0)

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _polygon_area(quad_pts: np.ndarray) -> float:
    """Vectorized Shoelace area of a 4-point polygon."""
    x = quad_pts[:, 0].astype(np.float64)
    y = quad_pts[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _signed_area(pts: Sequence[Tuple[int, int]]) -> float:
    """Signed area (positive = clockwise in screen Y-down coords)."""
    total = 0.0
    for idx in range(len(pts)):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % len(pts)]
        total += float(x1 * y2 - x2 * y1)
    return 0.5 * total


def _extract_hand_thumb_index_px(
    hand_entry: object,
    frame_w: int,
    frame_h: int,
    fallback_label: str = "",
) -> Optional[HandPortalPoints]:
    """Extract ONLY Thumb Tip (4) and Index Tip (8) pixel coords without clamping.

    Accepts `(landmarks, label)` tuples, `HandTelemetry`, or raw landmark lists.
    Returns `None` if landmarks are missing or contain non-finite coordinates.
    """
    if isinstance(hand_entry, HandPortalPoints):
        return hand_entry

    if isinstance(hand_entry, HandTelemetry):
        return HandPortalPoints(
            hand_label=hand_entry.hand_label or fallback_label,
            thumb_tip_px=hand_entry.thumb_tip_px,
            index_tip_px=hand_entry.index_tip_px,
        )

    hand_label = fallback_label
    hand_obj = hand_entry
    if isinstance(hand_entry, tuple) and len(hand_entry) == 2 and isinstance(hand_entry[1], str):
        hand_obj, hand_label = hand_entry[0], hand_entry[1]

    landmarks = hand_obj.landmark if hasattr(hand_obj, "landmark") else hand_obj
    if landmarks is None or len(landmarks) <= 8:
        return None

    lm_thumb = landmarks[4]
    lm_index = landmarks[8]
    tx = float(lm_thumb.x) if hasattr(lm_thumb, "x") else float(lm_thumb[0])
    ty = float(lm_thumb.y) if hasattr(lm_thumb, "y") else float(lm_thumb[1])
    ix = float(lm_index.x) if hasattr(lm_index, "x") else float(lm_index[0])
    iy = float(lm_index.y) if hasattr(lm_index, "y") else float(lm_index[1])

    if not (math.isfinite(tx) and math.isfinite(ty) and math.isfinite(ix) and math.isfinite(iy)):
        return None

    # Allow points near or slightly outside camera bounds without artificial clamps
    p_thumb = (int(round(tx * frame_w)), int(round(ty * frame_h)))
    p_index = (int(round(ix * frame_w)), int(round(iy * frame_h)))
    return HandPortalPoints(
        hand_label=hand_label,
        thumb_tip_px=p_thumb,
        index_tip_px=p_index,
    )


class PortalGeometryEngine:
    """Independent geometry engine computing valid 4-point portal polygons."""

    @staticmethod
    def resolve_left_right_hands(
        hand_points: Sequence[HandPortalPoints],
    ) -> Optional[Tuple[HandPortalPoints, HandPortalPoints]]:
        """Resolve `(left_hand, right_hand)` via MediaPipe handedness, never array order.

        If MediaPipe assigns ambiguous/identical handedness to two distinct hands in a
        frame, falls back to horizontal thumb/index centroid X ordering so hands never swap
        silently or overwrite one another.
        """
        if len(hand_points) < 2:
            return None

        h0, h1 = hand_points[0], hand_points[1]
        l0 = h0.hand_label.strip().lower()
        l1 = h1.hand_label.strip().lower()

        if l0.startswith("left") and l1.startswith("right"):
            return h0, h1
        if l0.startswith("right") and l1.startswith("left"):
            return h1, h0

        # Fallback disambiguation when both hands carry identical or missing labels
        cx0 = 0.5 * (h0.thumb_tip_px[0] + h0.index_tip_px[0])
        cx1 = 0.5 * (h1.thumb_tip_px[0] + h1.index_tip_px[0])
        return (h0, h1) if cx0 <= cx1 else (h1, h0)

    @staticmethod
    def order_and_untwist_quad(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        p3: Tuple[int, int],
        p4: Tuple[int, int],
    ) -> np.ndarray:
        """Order 4 points into a twist-free clockwise polygon `[TL, TR, BR, BL]`."""
        points = np.array([p1, p2, p3, p4], dtype=np.int32)
        # Convex hull prevents crossings even for strongly rotated hand poses.
        hull = cv2.convexHull(points, returnPoints=True).reshape(-1, 2)
        if len(hull) != 4:
            # Degenerate configurations are rejected by compute_geometry's area checks.
            return points
        centroid = hull.astype(np.float64).mean(axis=0)
        angles = np.arctan2(hull[:, 1] - centroid[1], hull[:, 0] - centroid[0])
        ordered = hull[np.argsort(angles)]
        # Start at the visually upper-left corner; preserve cyclic winding.
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start, axis=0)
        if _signed_area([tuple(map(int, p)) for p in ordered]) < 0:
            ordered = np.concatenate([ordered[:1], ordered[:0:-1]])
        return np.ascontiguousarray(ordered, dtype=np.int32)

    def compute_geometry(
        self,
        hand_entries: Sequence[object],
        frame_w: int,
        frame_h: int,
    ) -> PortalQuadGeometry:
        """Compute validated 4-point portal geometry strictly from Thumb (4) + Index (8)."""
        if len(hand_entries) < 2:
            return PortalQuadGeometry(
                is_valid=False,
                p1_left_thumb=None,
                p2_left_index=None,
                p3_right_index=None,
                p4_right_thumb=None,
                quad=None,
                bbox=(0, 0, 0, 0),
                area=0.0,
                rejection_reason="MISSING_HAND",
            )

        extracted: List[HandPortalPoints] = []
        for idx, entry in enumerate(hand_entries):
            pts = _extract_hand_thumb_index_px(entry, frame_w, frame_h, fallback_label=f"Hand{idx}")
            if pts is not None:
                extracted.append(pts)
            if len(extracted) == 2:
                break

        resolved = self.resolve_left_right_hands(extracted)
        if resolved is None:
            return PortalQuadGeometry(
                is_valid=False,
                p1_left_thumb=None,
                p2_left_index=None,
                p3_right_index=None,
                p4_right_thumb=None,
                quad=None,
                bbox=(0, 0, 0, 0),
                area=0.0,
                rejection_reason="INVALID_COORDS",
            )

        left_hand, right_hand = resolved
        p1 = left_hand.thumb_tip_px
        p2 = left_hand.index_tip_px
        p3 = right_hand.index_tip_px
        p4 = right_hand.thumb_tip_px

        quad = self.order_and_untwist_quad(p1, p2, p3, p4)
        area = _polygon_area(quad)
        if area < 1.0:
            return PortalQuadGeometry(
                is_valid=False,
                p1_left_thumb=p1,
                p2_left_index=p2,
                p3_right_index=p3,
                p4_right_thumb=p4,
                quad=None,
                bbox=(0, 0, 0, 0),
                area=area,
                rejection_reason="ZERO_AREA",
            )

        x1 = max(0, int(np.min(quad[:, 0])))
        y1 = max(0, int(np.min(quad[:, 1])))
        x2 = min(frame_w, int(np.max(quad[:, 0])) + 1)
        y2 = min(frame_h, int(np.max(quad[:, 1])) + 1)
        if x2 <= x1 or y2 <= y1:
            return PortalQuadGeometry(
                is_valid=False,
                p1_left_thumb=p1,
                p2_left_index=p2,
                p3_right_index=p3,
                p4_right_thumb=p4,
                quad=quad,
                bbox=(0, 0, 0, 0),
                area=area,
                rejection_reason="OUT_OF_BOUNDS",
            )

        return PortalQuadGeometry(
            is_valid=True,
            p1_left_thumb=p1,
            p2_left_index=p2,
            p3_right_index=p3,
            p4_right_thumb=p4,
            quad=quad,
            bbox=(x1, y1, x2, y2),
            area=area,
            rejection_reason="",
        )


class MarvelPortalRenderer:
    """High-performance ROI Marvel Comic-Book Portal & minimal HUD renderer."""

    def __init__(self, assets_dir: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.assets_dir: Path = assets_dir if assets_dir is not None else (base_dir / "assets")
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        self.characters: List[MarvelCharacterProfile] = _default_profiles()
        self.active_index: int = 0
        self._shockwaves: List[_ShockwaveEffect] = []
        self._snap_flash_frames: int = 0
        self._last_switch_source: str = ""

        # Precomputed static shader lookup tables & comic halftone dot matrix
        self._hero_luts: Dict[str, np.ndarray] = {
            char.slug: _build_hero_lut(char.bgr_tint) for char in self.characters
        }
        self._halftone_dots: np.ndarray = _build_halftone_matrix(max_h=1080, max_w=1920, cell=6)

        self.geometry_engine = PortalGeometryEngine()
        self.face_renderer = MarvelFaceFilterRenderer(assets_dir=self.assets_dir)

    @property
    def active_character(self) -> MarvelCharacterProfile:
        """Return the currently selected Marvel character profile."""
        return self.characters[self.active_index]

    def switch_character(self, step: int = 1, source_label: str = "") -> MarvelCharacterProfile:
        """Advance to the next Marvel comic dimension and arm the HUD flash banner."""
        self.active_index = (self.active_index + step) % len(self.characters)
        self._snap_flash_frames = 18
        self._last_switch_source = source_label
        return self.active_character

    def reset_character(self) -> MarvelCharacterProfile:
        """Reset back to the first Marvel comic dimension in the roster."""
        self.active_index = 0
        self._snap_flash_frames = 0
        return self.active_character

    def trigger_shockwave(
        self,
        midpoint_px: Tuple[int, int],
        accent_bgr: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Spawn a 6-frame expanding neon shockwave circle at the pinch midpoint."""
        color = accent_bgr if accent_bgr is not None else self.active_character.accent_bgr
        self._shockwaves.append(
            _ShockwaveEffect(center=midpoint_px, accent_bgr=color, frame_index=0, total_frames=6)
        )

    @staticmethod
    def order_portal_quad(*pts_args: object) -> np.ndarray:
        """Order 4 fingertip points into a twist-free `[TL, TR, BR, BL]` quadrilateral."""
        if len(pts_args) == 2:
            pair_a, pair_b = pts_args[0], pts_args[1]
            raw_pts = [pair_a[0], pair_a[1], pair_b[0], pair_b[1]]
        elif len(pts_args) == 4:
            raw_pts = list(pts_args)
        else:
            raise ValueError(f"Expected 2 pairs or 4 points, got {len(pts_args)} arguments")

        pts_int = [(int(pt[0]), int(pt[1])) for pt in raw_pts]
        return PortalGeometryEngine.order_and_untwist_quad(
            pts_int[0], pts_int[1], pts_int[2], pts_int[3]
        )

    def _apply_comic_world_shader(
        self,
        roi_bgr: np.ndarray,
        slug: str,
        roi_origin: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Transform only `roi_bgr` into a crisp Marvel comic panel via cached LUT & ink edges."""
        rh, rw = roi_bgr.shape[:2]
        if rh == 0 or rw == 0:
            return roi_bgr

        # 1. Fast 0.5x bilateral smoothing -> precomputed uint8 hero Color LUT -> upscale
        small = cv2.resize(roi_bgr, (max(1, rw // 2), max(1, rh // 2)), interpolation=cv2.INTER_LINEAR)
        filtered = cv2.bilateralFilter(small, d=5, sigmaColor=65, sigmaSpace=65)
        lut = self._hero_luts.get(slug, self._hero_luts["spider_man"])
        quantized_small = cv2.LUT(filtered, lut)
        comic_roi = cv2.resize(quantized_small, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # 2. Apply cached 3-channel Ben-Day halftone pattern in ROI coordinates (zero allocation)
        x1, y1 = roi_origin
        ht_h, ht_w = self._halftone_dots.shape[:2]
        if y1 + rh <= ht_h and x1 + rw <= ht_w:
            cv2.subtract(comic_roi, self._halftone_dots[y1 : y1 + rh, x1 : x1 + rw], dst=comic_roi)

        # 3. Bold black comic ink contours while preserving readable facial features
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
        )
        return cv2.bitwise_and(comic_roi, comic_roi, mask=edges)

    def render_ar_portal_scene(
        self,
        base_frame: np.ndarray,
        hand_landmarks_list: Sequence[object],
        face_landmarks_list: Optional[Sequence[object]] = None,
        portal_geometry: Optional[PortalQuadGeometry] = None,
        in_place: bool = False,
        timestamp: Optional[float] = None,
        timings: Optional[PipelineTimings] = None,
    ) -> Tuple[np.ndarray, str]:
        """Render the Marvel Comic-Book Portal strictly inside the 4-point quad ROI.

        - `< 2` hands or invalid quad: leaves `base_frame` 100% untouched.
        - Valid 2-hand quad: applies cel LUT + halftone + ink edges (plus optional comic
          face filter) strictly inside `quad` via an ROI mask without copying the whole frame
          when `in_place=True`.
        """
        out = base_frame if in_place else base_frame.copy()
        frame_h, frame_w = out.shape[:2]

        t_g0 = time.perf_counter()
        geom = (
            portal_geometry
            if portal_geometry is not None
            else self.geometry_engine.compute_geometry(hand_landmarks_list, frame_w, frame_h)
        )
        if timings is not None and portal_geometry is None:
            timings.portal_geom_ms = (time.perf_counter() - t_g0) * 1000.0

        if len(hand_landmarks_list) < 2:
            mode_label = "HANDS: 1/2" if len(hand_landmarks_list) == 1 else "HANDS: 0/2"
            return out, mode_label

        if not geom.is_valid or geom.quad is None:
            return out, f"HANDS: 2/2 [{geom.rejection_reason or 'INVALID'}]"

        quad = geom.quad
        x1, y1, x2, y2 = geom.bbox
        rw, rh = x2 - x1, y2 - y1
        char = self.active_character

        if rw >= 4 and rh >= 4:
            roi = out[y1:y2, x1:x2]
            t_c0 = time.perf_counter()
            comic_roi = self._apply_comic_world_shader(roi, char.slug, roi_origin=(x1, y1))
            if timings is not None:
                timings.comic_shader_ms = (time.perf_counter() - t_c0) * 1000.0

            if face_landmarks_list:
                t_f0 = time.perf_counter()
                self.face_renderer.apply_face_filter(
                    frame=comic_roi,
                    face_landmarks_list=face_landmarks_list,
                    character_slug=char.slug,
                    accent_bgr=char.accent_bgr,
                    secondary_bgr=char.secondary_bgr,
                    timestamp=timestamp,
                    full_frame_wh=(frame_w, frame_h),
                    roi_offset_xy=(x1, y1),
                    in_place=True,
                )
                if timings is not None:
                    timings.face_filter_ms = (time.perf_counter() - t_f0) * 1000.0

            mask = np.zeros((rh, rw), dtype=np.uint8)
            local_quad = quad - np.array([x1, y1], dtype=np.int32)
            cv2.fillPoly(mask, [local_quad], 255)
            cv2.copyTo(comic_roi, mask, out[y1:y2, x1:x2])

            if timings is not None:
                rx, ry, rbw, rbh = cv2.boundingRect(mask)
                timings.rendered_bbox = (x1 + rx, y1 + ry, x1 + rx + rbw, y1 + ry + rbh)
                timings.dest_wh = (rw, rh)

        # Stroke portal border and visible fingertip vertices
        cv2.polylines(out, [quad], isClosed=True, color=CYAN_GLOW, thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(out, [quad], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for cx, cy in quad:
            ix, iy = int(cx), int(cy)
            if -8 <= ix < frame_w + 8 and -8 <= iy < frame_h + 8:
                cv2.circle(out, (ix, iy), 6, CYAN_GLOW, -1, cv2.LINE_AA)
                cv2.circle(out, (ix, iy), 3, (255, 255, 255), -1, cv2.LINE_AA)

        return out, "HANDS: 2/2 [PORTAL ACTIVE]"

    @staticmethod
    def render_fingertip_dots(
        frame: np.ndarray,
        hand_landmarks_list: Sequence[object],
    ) -> None:
        """Draw small cyan dots on Thumb Tip (4) and Index Tip (8) when < 2 hands are detected."""
        frame_h, frame_w = frame.shape[:2]
        for idx, hand_obj in enumerate(hand_landmarks_list):
            pts = _extract_hand_thumb_index_px(hand_obj, frame_w, frame_h, fallback_label=f"Hand{idx}")
            if pts is None:
                continue
            for pt in (pts.thumb_tip_px, pts.index_tip_px):
                cv2.circle(frame, pt, 6, CYAN_GLOW, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def render_shockwaves(self, frame: np.ndarray) -> None:
        """Advance and render active 6-frame shockwaves using ROI-only blending."""
        fh, fw = frame.shape[:2]
        surviving: List[_ShockwaveEffect] = []
        for fx in self._shockwaves:
            progress = (fx.frame_index + 1) / float(fx.total_frames)
            radius = int(16 + progress * 76)
            thickness = max(1, int(round(5 * (1.0 - progress * 0.7))))
            pad = radius + thickness + 4

            rx1 = max(0, fx.center[0] - pad)
            ry1 = max(0, fx.center[1] - pad)
            rx2 = min(fw, fx.center[0] + pad)
            ry2 = min(fh, fx.center[1] + pad)
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                overlay = roi.copy()
                local_c = (fx.center[0] - rx1, fx.center[1] - ry1)
                cv2.circle(overlay, local_c, radius, fx.accent_bgr, thickness, cv2.LINE_AA)
                cv2.circle(overlay, local_c, max(4, int(radius * 0.58)), NEON_ORANGE, 2, cv2.LINE_AA)
                alpha = max(0.18, 0.85 * (1.0 - progress * 0.65))
                cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)

            fx.frame_index += 1
            if fx.frame_index < fx.total_frames:
                surviving.append(fx)
        self._shockwaves = surviving

    def render_hud(
        self,
        frame: np.ndarray,
        os_backend_badge: str,
        fps: float,
        portal_mode: str,
        telemetries: Sequence[HandTelemetry],
        diagnostics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Draw a minimal dark-mode (#121212) top/bottom strip with ROI-only blending."""
        frame_h, frame_w = frame.shape[:2]
        char = self.active_character
        compact = frame_w < 920

        top_h = 38
        bot_h = 44

        # ROI-only dark panel blending (avoids copying the 1280x720 center canvas)
        if frame_h > top_h:
            top_strip = frame[0:top_h, :]
            top_strip[:] = (top_strip.astype(np.uint16) * 28 + 18 * 72) // 100
        if frame_h > bot_h:
            bot_strip = frame[frame_h - bot_h : frame_h, :]
            bot_strip[:] = (bot_strip.astype(np.uint16) * 28 + 18 * 72) // 100

        cv2.line(frame, (0, top_h), (frame_w, top_h), char.accent_bgr, 1, cv2.LINE_AA)
        cv2.line(frame, (0, frame_h - bot_h), (frame_w, frame_h - bot_h), char.accent_bgr, 1, cv2.LINE_AA)

        # --- MINIMAL TOP STRIP ---
        cv2.putText(
            frame,
            os_backend_badge,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38 if compact else 0.44,
            NEON_GREEN,
            1,
            cv2.LINE_AA,
        )

        char_x = 104 if compact else 128
        cv2.rectangle(frame, (char_x, 9), (char_x + 4, 29), char.accent_bgr, -1)
        cv2.putText(
            frame,
            f"{char.name} [{self.active_index + 1}/{len(self.characters)}]",
            (char_x + 10, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46 if compact else 0.56,
            TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

        if "2/2" in portal_mode and "PORTAL ACTIVE" in portal_mode:
            mode_color = NEON_GREEN
        elif "1/2" in portal_mode:
            mode_color = (0, 230, 255)
        else:
            mode_color = NEON_ORANGE

        fps_text = f"FPS {fps:4.1f}"
        fps_x = frame_w - (82 if compact else 98)
        cv2.putText(frame, fps_text, (fps_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40 if compact else 0.46, TEXT_WHITE, 1, cv2.LINE_AA)

        mode_scale = 0.32 if compact else 0.41
        (mode_w, _), _ = cv2.getTextSize(portal_mode, cv2.FONT_HERSHEY_SIMPLEX, mode_scale, 1)
        mode_x = max(char_x + 185, fps_x - mode_w - 16)
        cv2.putText(frame, portal_mode, (mode_x, 25), cv2.FONT_HERSHEY_SIMPLEX, mode_scale, mode_color, 1, cv2.LINE_AA)

        # --- MINIMAL BOTTOM STRIP ---
        bar_top = frame_h - bot_h
        held_hands = [item for item in telemetries if item.state == "PINCH_HELD"]
        primary: Optional[HandTelemetry] = (
            held_hands[0]
            if held_hands
            else (min(telemetries, key=lambda item: item.smoothed_ratio) if telemetries else None)
        )

        ratio_val = primary.smoothed_ratio if primary is not None else 0.0
        max_gauge_ratio = 0.80
        gauge_x = 12
        gauge_y = bar_top + 16
        gauge_w = max(140, min(200 if compact else 280, int(frame_w * 0.24)))
        gauge_h = 12

        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (34, 36, 42), -1)
        if primary is not None:
            fill_w = int(round(gauge_w * float(np.clip(ratio_val / max_gauge_ratio, 0.0, 1.0))))
            bar_color = NEON_ORANGE if ratio_val < 0.29 else (NEON_GREEN if ratio_val > 0.38 else (0, 230, 255))
            cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), bar_color, -1)

        trig_x = gauge_x + int(round(gauge_w * (0.29 / max_gauge_ratio)))
        rel_x = gauge_x + int(round(gauge_w * (0.38 / max_gauge_ratio)))
        cv2.line(frame, (trig_x, gauge_y - 2), (trig_x, gauge_y + gauge_h + 2), NEON_ORANGE, 2)
        cv2.line(frame, (rel_x, gauge_y - 2), (rel_x, gauge_y + gauge_h + 2), NEON_GREEN, 2)
        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (82, 88, 98), 1)

        ratio_text = f"R={ratio_val:.2f}" if primary is not None else "R=--"
        hint_suffix = "[S][V][F][P][R][Q]" if compact else "[S]SNAP [V]REC [F]FACE [P]PERF [R]RST [Q]EXIT"
        cv2.putText(
            frame,
            f"{ratio_text} | {hint_suffix}",
            (gauge_x + gauge_w + 10, bar_top + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33 if compact else 0.38,
            TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Snap Readiness Status Pill
        if self._snap_flash_frames > 0:
            self._snap_flash_frames -= 1
            status_text = "SWITCHED!"
            status_color = NEON_ORANGE
            box_fill = (18, 42, 72)
        elif primary is None:
            status_text = "NO HAND"
            status_color = TEXT_MUTED
            box_fill = CHARCOAL_PANEL
        elif primary.state == "PINCH_HELD":
            status_text = "CHARGING" if compact else "CHARGING (RELEASE)"
            status_color = (0, 230, 255)
            box_fill = (36, 36, 18)
        elif not primary.is_index_extended:
            status_text = "FIST GUARD"
            status_color = (0, 195, 255)
            box_fill = CHARCOAL_PANEL
        else:
            status_text = "READY"
            status_color = NEON_GREEN
            box_fill = CHARCOAL_PANEL

        font_scale = 0.40 if compact else 0.45
        thickness = 2 if self._snap_flash_frames > 0 else 1
        (text_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        status_w = max(124 if compact else 150, text_w + 26)
        status_x = frame_w - status_w - 12
        status_y1 = bar_top + 8
        status_y2 = frame_h - 8
        cv2.rectangle(frame, (status_x, status_y1), (status_x + status_w, status_y2), box_fill, -1)
        cv2.rectangle(frame, (status_x, status_y1), (status_x + status_w, status_y2), status_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            status_text,
            (status_x + 13, status_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            status_color,
            thickness,
            cv2.LINE_AA,
        )

        # Optional lightweight performance diagnostics card (toggled via [P] or --profile)
        if diagnostics:
            lines = [
                f"FPS Avg:{diagnostics.get('avg_fps', fps):.1f} Min:{diagnostics.get('min_fps', fps):.1f} Drop:{int(diagnostics.get('dropped_frames', 0))}",
                f"Frame:{diagnostics.get('total_frame_ms', 0.0):.1f}ms Hand:{diagnostics.get('hand_detect_ms', 0.0):.1f}ms Render:{diagnostics.get('portal_render_ms', 0.0):.1f}ms",
                f"Face:{diagnostics.get('face_detect_ms', 0.0):.1f}ms Geom:{diagnostics.get('portal_geom_ms', 0.0):.2f}ms Rec:{diagnostics.get('record_ms', 0.0):.2f}ms",
            ]
            pw = min(390, frame_w - 24)
            ph = 18 * len(lines) + 12
            px1, py1 = 12, top_h + 8
            px2, py2 = px1 + pw, min(frame_h - bot_h - 4, py1 + ph)
            if py2 > py1:
                panel_roi = frame[py1:py2, px1:px2]
                panel_roi[:] = (panel_roi.astype(np.uint16) * 25 + 14 * 75) // 100
                cv2.rectangle(frame, (px1, py1), (px2, py2), char.accent_bgr, 1, cv2.LINE_AA)
                for row_idx, line in enumerate(lines):
                    cv2.putText(
                        frame,
                        line,
                        (px1 + 8, py1 + 17 + row_idx * 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.41,
                        (225, 238, 248),
                        1,
                        cv2.LINE_AA,
                    )
