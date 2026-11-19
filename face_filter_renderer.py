"""Ultra-fast (< 1 ms) vectorized Marvel comic-book face overlay renderer.

Draws bold cel-shaded comic character masks and ink outlines directly onto the
468 facial landmarks using pure OpenCV polygon & vector primitives with zero
slow texture generation or per-triangle Delaunay warping loops:
- spider_man: Red cel-shaded mask, sharp nasal-radiating black web lines, and
  blink-tracking black-framed white comic lenses.
- scarlet_witch: Stylized crimson comic tiara/crown with black ink outlines,
  glowing red comic eyes, and floating red comic energy sparks.
- doctor_strange: Stylized forehead Third Eye with comic ink contours and
  rotating geometric golden mandala rings.
- iron_man: Bold red & gold cel-shaded comic faceplate with black ink armor
  seams and glowing cyan comic eye slits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


FACE_OVAL_INDICES: Tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
TOP_OVAL_INDICES = {10, 338, 297, 332, 284, 21, 54, 103, 67, 109}

LEFT_EYE_INDICES: Tuple[int, ...] = (
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
)
RIGHT_EYE_INDICES: Tuple[int, ...] = (
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
)


@dataclass(frozen=True)
class FaceGeometry:
    """Lightweight 2D pixel-space facial geometry and expression ratios."""

    points: np.ndarray            # (N, 2) float32
    extended_oval: np.ndarray     # (36, 2) int32
    face_center: np.ndarray       # (2,) float32
    face_width: float
    face_height: float
    up_vec: np.ndarray            # (2,) float32 chin -> forehead
    right_vec: np.ndarray         # (2,) float32 left_cheek -> right_cheek
    roll_deg: float
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    left_eye_open: float
    right_eye_open: float
    mouth_open_ratio: float
    brow_raise_ratio: float


def _landmarks_to_px(
    face_landmarks: object,
    frame_w: int,
    frame_h: int,
    roi_offset_xy: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Convert normalized MediaPipe landmarks into (N, 2) pixel coordinates in ROI space."""
    ox, oy = float(roi_offset_xy[0]), float(roi_offset_xy[1])
    lms = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    coords = [
        (float(lm.x) * frame_w - ox, float(lm.y) * frame_h - oy)
        if hasattr(lm, "x")
        else (float(lm[0]) * frame_w - ox, float(lm[1]) * frame_h - oy)
        for lm in lms
    ]
    return np.array(coords, dtype=np.float32)


def _extract_face_geometry(
    face_landmarks: object,
    frame_w: int,
    frame_h: int,
    roi_offset_xy: Tuple[int, int] = (0, 0),
) -> Optional[FaceGeometry]:
    """Extract fast facial basis vectors, cranium-extended oval, and blink ratios."""
    pts = _landmarks_to_px(face_landmarks, frame_w, frame_h, roi_offset_xy=roi_offset_xy)
    if len(pts) < 468:
        return None

    up_delta = pts[10] - pts[152]
    face_height = max(float(np.linalg.norm(up_delta)), 1e-3)
    up_vec = (up_delta / face_height).astype(np.float32)

    right_delta = pts[454] - pts[234]
    face_width = max(float(np.linalg.norm(right_delta)), 1e-3)
    right_vec = (right_delta / face_width).astype(np.float32)
    roll_deg = math.degrees(math.atan2(float(right_vec[1]), float(right_vec[0])))

    oval_pts: List[np.ndarray] = []
    for idx in FACE_OVAL_INDICES:
        pt = pts[idx].copy()
        if idx in TOP_OVAL_INDICES:
            lift = 0.13 if idx in (10, 109, 338) else 0.08
            pt = pt + up_vec * (face_height * lift)
        oval_pts.append(pt)
    extended_oval = np.round(np.array(oval_pts, dtype=np.float32)).astype(np.int32)

    l_w = max(float(np.linalg.norm(pts[133] - pts[33])), 1e-3)
    l_h = float(np.linalg.norm(pts[159] - pts[145]))
    left_eye_open = float(np.clip((l_h / l_w) / 0.28, 0.18, 1.15))

    r_w = max(float(np.linalg.norm(pts[263] - pts[362])), 1e-3)
    r_h = float(np.linalg.norm(pts[386] - pts[374]))
    right_eye_open = float(np.clip((r_h / r_w) / 0.28, 0.18, 1.15))

    mouth_w = max(float(np.linalg.norm(pts[291] - pts[61])), 1e-3)
    mouth_h = float(np.linalg.norm(pts[14] - pts[13]))
    mouth_open_ratio = float(np.clip(((mouth_h / mouth_w) - 0.06) / 0.28, 0.0, 1.5))

    l_brow_gap = float(np.dot(pts[105] - pts[159], up_vec))
    r_brow_gap = float(np.dot(pts[334] - pts[386], up_vec))
    brow_norm = (0.5 * (l_brow_gap + r_brow_gap)) / max(face_height, 1e-3)
    brow_raise_ratio = float(np.clip((brow_norm - 0.092) / 0.048, 0.0, 1.5))

    if len(pts) > 473:
        left_eye_center = pts[468]
        right_eye_center = pts[473]
    else:
        left_eye_center = np.mean(pts[list(LEFT_EYE_INDICES)], axis=0)
        right_eye_center = np.mean(pts[list(RIGHT_EYE_INDICES)], axis=0)

    face_center = 0.5 * (pts[10] + pts[152])

    return FaceGeometry(
        points=pts,
        extended_oval=extended_oval,
        face_center=face_center,
        face_width=face_width,
        face_height=face_height,
        up_vec=up_vec,
        right_vec=right_vec,
        roll_deg=roll_deg,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        left_eye_open=left_eye_open,
        right_eye_open=right_eye_open,
        mouth_open_ratio=mouth_open_ratio,
        brow_raise_ratio=brow_raise_ratio,
    )


class MarvelFaceFilterRenderer:
    """Fast comic-book face overlay renderer using pure polygon & ink primitives."""

    def __init__(self, assets_dir: Optional[Path] = None) -> None:
        self.assets_dir = assets_dir
        self._mandala_angle: float = 0.0
        self._last_strange_ts: Optional[float] = None

    def apply_face_filter(
        self,
        frame: np.ndarray,
        face_landmarks_list: Sequence[object],
        character_slug: str,
        accent_bgr: Tuple[int, int, int],
        secondary_bgr: Tuple[int, int, int],
        timestamp: Optional[float] = None,
        full_frame_wh: Optional[Tuple[int, int]] = None,
        roi_offset_xy: Tuple[int, int] = (0, 0),
        in_place: bool = False,
    ) -> np.ndarray:
        """Render the stylized comic-book hero mask onto faces inside `frame` (or cropped ROI)."""
        out = frame if in_place else frame.copy()
        if not face_landmarks_list:
            return out

        ts = time.perf_counter() if timestamp is None else float(timestamp)
        roi_h, roi_w = out.shape[:2]
        full_w, full_h = full_frame_wh if full_frame_wh is not None else (roi_w, roi_h)

        for face_lms in face_landmarks_list:
            geom = _extract_face_geometry(face_lms, full_w, full_h, roi_offset_xy=roi_offset_xy)
            if geom is None:
                continue

            # Skip faces completely outside the cropped portal ROI
            cx, cy = float(geom.face_center[0]), float(geom.face_center[1])
            if (
                cx < -geom.face_width * 0.75
                or cx > roi_w + geom.face_width * 0.75
                or cy < -geom.face_height * 0.75
                or cy > roi_h + geom.face_height * 0.75
            ):
                continue

            if character_slug == "spider_man":
                self._draw_comic_spider_man(out, geom)
            elif character_slug == "scarlet_witch":
                self._draw_comic_scarlet_witch(out, geom, ts)
            elif character_slug == "doctor_strange":
                self._draw_comic_doctor_strange(out, geom, ts, accent_bgr, secondary_bgr)
            elif character_slug == "iron_man":
                self._draw_comic_iron_man(out, geom, ts)
            elif character_slug in {"thor", "loki", "black_panther", "deadpool", "captain_america", "wolverine", "venom", "ghost_rider"}:
                self._draw_extra_hero(out, geom, character_slug, ts)
            else:
                self._draw_comic_spider_man(out, geom)

        return out

    # ------------------------------------------------------------------
    # 1. SPIDER-MAN: Red cel-shaded comic mask + web lines + blink eyes
    # ------------------------------------------------------------------
    def _draw_comic_spider_man(self, canvas: np.ndarray, geom: FaceGeometry) -> None:
        # Crisp two-tone comic crimson fill + bold ink outline
        mask_layer = canvas.copy()
        cv2.fillPoly(mask_layer, [geom.extended_oval], (28, 28, 224), cv2.LINE_AA)
        # Subtle comic cel shadow on right jaw/cheek half
        shadow_oval = np.vstack([geom.extended_oval[:19], geom.points[[152, 13, 1, 6, 9, 10]].astype(np.int32)])
        cv2.fillPoly(mask_layer, [shadow_oval], (20, 20, 178), cv2.LINE_AA)
        cv2.addWeighted(mask_layer, 0.86, canvas, 0.14, 0.0, canvas)

        # Sharp black radial + concentric web-lines clipped cleanly inside mask oval
        nose = geom.points[6]
        oval_f = geom.extended_oval.astype(np.float32)
        spoke_idx = np.linspace(0, len(oval_f) - 1, 14, endpoint=False, dtype=int)
        spokes = [oval_f[i] for i in spoke_idx]
        ink_w = max(2, int(round(geom.face_width * 0.012)))
        ink = (12, 12, 16)

        p_nose = (int(round(nose[0])), int(round(nose[1])))
        for pt in spokes:
            cv2.line(canvas, p_nose, (int(round(pt[0])), int(round(pt[1]))), ink, ink_w, cv2.LINE_AA)

        for t in (0.20, 0.39, 0.58, 0.78, 0.94):
            for idx in range(len(spokes)):
                pa = nose + (spokes[idx] - nose) * t
                pb = nose + (spokes[(idx + 1) % len(spokes)] - nose) * t
                mid = 0.5 * (pa + pb)
                ctrl = mid + (nose - mid) * 0.16
                arc = np.round(np.array([pa, ctrl, pb], dtype=np.float32)).astype(np.int32)
                cv2.polylines(canvas, [arc], False, ink, ink_w, cv2.LINE_AA)

        cv2.polylines(canvas, [geom.extended_oval], True, ink, max(3, ink_w + 1), cv2.LINE_AA)

        # Blink-tracking white Spider-Man lenses with bold black comic frames
        for is_left in (True, False):
            self._draw_spidey_comic_lens(canvas, geom, is_left=is_left)

    @staticmethod
    def _draw_spidey_comic_lens(canvas: np.ndarray, geom: FaceGeometry, is_left: bool) -> None:
        pts = geom.points
        if is_left:
            inner_c, outer_c = pts[133], pts[33]
            brow_pt, cheek_pt = pts[105], pts[118]
            open_ratio, side = geom.left_eye_open, -1.0
        else:
            inner_c, outer_c = pts[362], pts[263]
            brow_pt, cheek_pt = pts[334], pts[347]
            open_ratio, side = geom.right_eye_open, 1.0

        eye_c = 0.5 * (inner_c + outer_c)
        eye_w = max(float(np.linalg.norm(outer_c - inner_c)), geom.face_width * 0.14)
        # Squinting narrows the white aperture while thickening the outer mechanical shutter frame
        squint_t = float(np.clip((0.78 - open_ratio) / 0.45, 0.0, 1.0))
        v_scale = max(0.18, 0.24 + 0.80 * float(np.clip(open_ratio, 0.18, 1.15)) - 0.10 * squint_t)
        r_vec, u_vec = geom.right_vec, geom.up_vec

        p_in = eye_c - r_vec * (side * eye_w * 0.56) - u_vec * (eye_w * 0.10)
        p_tip = eye_c + r_vec * (side * eye_w * 1.12) + u_vec * (eye_w * 0.78 * v_scale)
        p_top = 0.5 * (eye_c + brow_pt) + r_vec * (side * eye_w * 0.25) + u_vec * (eye_w * 0.46 * v_scale)
        p_lo = eye_c + r_vec * (side * eye_w * 0.64) - u_vec * (eye_w * 0.24 * v_scale)
        p_li = 0.5 * (eye_c + cheek_pt) - r_vec * (side * eye_w * 0.18) - u_vec * (eye_w * 0.28 * v_scale)

        outer_poly = np.round(np.array([p_in, p_top, p_tip, p_lo, p_li], dtype=np.float32)).astype(np.int32)
        cent = np.mean(outer_poly.astype(np.float32), axis=0)
        inner_scale = 0.70 - 0.22 * squint_t
        inner_poly = np.round(cent + (outer_poly.astype(np.float32) - cent) * inner_scale).astype(np.int32)

        cv2.fillPoly(canvas, [outer_poly], (10, 10, 14), cv2.LINE_AA)
        if squint_t > 0.05:
            shutter_w = max(2, int(round(eye_w * (0.06 + 0.11 * squint_t))))
            cv2.polylines(canvas, [outer_poly], True, (8, 8, 12), shutter_w, cv2.LINE_AA)
        cv2.fillPoly(canvas, [inner_poly], (248, 250, 255), cv2.LINE_AA)
        cv2.polylines(canvas, [inner_poly], True, (18, 18, 24), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 2. SCARLET WITCH: Comic crown/tiara + glowing red eyes + sparks
    # ------------------------------------------------------------------
    def _draw_comic_scarlet_witch(self, canvas: np.ndarray, geom: FaceGeometry, ts: float) -> None:
        pts = geom.points
        r_vec, u_vec = geom.right_vec, geom.up_vec
        w, h = geom.face_width, geom.face_height
        chaos_surge = float(np.clip(max(geom.mouth_open_ratio, geom.brow_raise_ratio), 0.0, 1.5))
        peak_pt = pts[10] + u_vec * (h * 0.28)

        # Stylized crimson Scarlet Witch comic tiara on the forehead
        crown_poly = np.round(
            np.array(
                [
                    pts[127] - r_vec * (w * 0.04) + u_vec * (h * 0.04),
                    pts[21] - r_vec * (w * 0.06) + u_vec * (h * 0.24),
                    pts[67] + u_vec * (h * 0.14),
                    peak_pt,
                    pts[297] + u_vec * (h * 0.14),
                    pts[251] + r_vec * (w * 0.06) + u_vec * (h * 0.24),
                    pts[356] + r_vec * (w * 0.04) + u_vec * (h * 0.04),
                    pts[300] + u_vec * (h * 0.03),
                    pts[334] + u_vec * (h * 0.02),
                    pts[9] + u_vec * (h * 0.02),
                    pts[105] + u_vec * (h * 0.02),
                    pts[70] + u_vec * (h * 0.03),
                ],
                dtype=np.float32,
            )
        ).astype(np.int32)
        cv2.fillPoly(canvas, [crown_poly], (36, 22, 188), cv2.LINE_AA)
        cv2.polylines(canvas, [crown_poly], True, (12, 10, 16), 3, cv2.LINE_AA)

        # Center crown comic diamond facet + crown-peak surge aura on mouth open / brow raise
        gem_c = pts[151] + u_vec * (h * 0.09)
        gem = np.round(
            np.array(
                [gem_c + u_vec * (h * 0.07), gem_c + r_vec * (w * 0.035), gem_c - u_vec * (h * 0.04), gem_c - r_vec * (w * 0.035)],
                dtype=np.float32,
            )
        ).astype(np.int32)
        cv2.fillPoly(canvas, [gem], (95, 55, 255), cv2.LINE_AA)
        cv2.polylines(canvas, [gem], True, (14, 10, 18), 2, cv2.LINE_AA)

        if chaos_surge > 0.18:
            pkx, pky = int(round(peak_pt[0])), int(round(peak_pt[1]))
            peak_r = max(6, int(round(w * (0.06 + 0.08 * chaos_surge))))
            cv2.circle(canvas, (pkx, pky), peak_r, (70, 35, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (pkx, pky), max(3, peak_r // 2), (185, 130, 255), -1, cv2.LINE_AA)
            for ray_idx in range(4):
                th = ts * 3.0 + ray_idx * (math.pi / 4.0)
                dx = int(round(math.cos(th) * peak_r * 1.4))
                dy = int(round(math.sin(th) * peak_r * 1.4))
                cv2.line(canvas, (pkx - dx, pky - dy), (pkx + dx, pky + dy), (110, 65, 255), 2, cv2.LINE_AA)

        # Glowing red comic eyes + intensified crimson Chaos Magic aura on expression surge
        for eye_indices, eye_c in ((LEFT_EYE_INDICES, geom.left_eye_center), (RIGHT_EYE_INDICES, geom.right_eye_center)):
            eye_poly = np.round(pts[list(eye_indices)]).astype(np.int32)
            cx, cy = int(round(eye_c[0])), int(round(eye_c[1]))
            if chaos_surge > 0.18:
                aura_r = max(6, int(round(w * (0.065 + 0.065 * chaos_surge))))
                cv2.circle(canvas, (cx, cy), aura_r, (45, 20, 235), 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), max(4, aura_r // 2), (95, 55, 255), 2, cv2.LINE_AA)
            cv2.fillPoly(canvas, [eye_poly], (45, 22, 245), cv2.LINE_AA)
            cv2.polylines(canvas, [eye_poly], True, (14, 10, 18), 2, cv2.LINE_AA)
            core_r = max(3, int(round(w * (0.030 + 0.014 * chaos_surge))))
            cv2.circle(canvas, (cx, cy), core_r, (180, 145, 255), -1, cv2.LINE_AA)

        # Floating comic red energy sparks around temples/cheeks
        spark_count = 10 if chaos_surge > 0.35 else 8
        for k in range(spark_count):
            phase = ts * (2.4 + 1.2 * chaos_surge) + k * 0.78
            sx = int(round(geom.face_center[0] + math.sin(phase + k) * w * 0.70))
            sy = int(round(geom.face_center[1] - ((phase * 45.0) % (h * 1.1)) + h * 0.45))
            r_spark = (7 if k % 2 == 0 else 5) + (2 if chaos_surge > 0.35 else 0)
            cv2.line(canvas, (sx - r_spark, sy), (sx + r_spark, sy), (90, 50, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (sx, sy - r_spark), (sx, sy + r_spark), (90, 50, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 2, (210, 180, 255), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 3. DOCTOR STRANGE: Comic Third Eye + rotating golden mandala rings
    # ------------------------------------------------------------------
    def _draw_comic_doctor_strange(
        self,
        canvas: np.ndarray,
        geom: FaceGeometry,
        ts: float,
        accent_bgr: Tuple[int, int, int],
        secondary_bgr: Tuple[int, int, int],
    ) -> None:
        pts = geom.points
        r_vec, u_vec = geom.right_vec, geom.up_vec
        eye_c = 0.55 * pts[151] + 0.45 * pts[9]
        mystic_surge = float(np.clip(geom.brow_raise_ratio, 0.0, 1.5))
        ew = max(12.0, geom.face_width * (0.16 + 0.025 * mystic_surge))
        eh = ew * (0.48 + 0.34 * mystic_surge)

        if self._last_strange_ts is None:
            self._mandala_angle = ts * 1.4
        else:
            dt = max(0.0, min(ts - self._last_strange_ts, 0.15))
            self._mandala_angle += dt * (1.4 + 3.4 * mystic_surge)
        self._last_strange_ts = ts

        tx, ty = int(round(eye_c[0])), int(round(eye_c[1]))
        if mystic_surge > 0.18:
            halo_r = max(10, int(round(ew * (0.95 + 0.45 * mystic_surge))))
            cv2.circle(canvas, (tx, ty), halo_r, accent_bgr, 2, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), max(6, int(round(halo_r * 0.72))), secondary_bgr, 1, cv2.LINE_AA)

        # Stylized Third Eye on forehead with bold black comic ink outline
        eyelid = np.round(
            np.array(
                [
                    eye_c - r_vec * ew,
                    eye_c - r_vec * (ew * 0.45) + u_vec * (eh * 0.9),
                    eye_c + u_vec * eh,
                    eye_c + r_vec * (ew * 0.45) + u_vec * (eh * 0.9),
                    eye_c + r_vec * ew,
                    eye_c + r_vec * (ew * 0.45) - u_vec * (eh * 0.9),
                    eye_c - u_vec * eh,
                    eye_c - r_vec * (ew * 0.45) - u_vec * (eh * 0.9),
                ],
                dtype=np.float32,
            )
        ).astype(np.int32)
        cv2.fillPoly(canvas, [eyelid], (24, 34, 52), cv2.LINE_AA)
        cv2.polylines(canvas, [eyelid], True, (12, 12, 16), 3, cv2.LINE_AA)

        ir = max(4, int(round(eh * 0.72)))
        iris_bgr = (140, 245, 255) if mystic_surge > 0.25 else (70, 225, 255)
        cv2.circle(canvas, (tx, ty), ir, iris_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), max(2, ir // 2), (140, 255, 165) if mystic_surge > 0.25 else (90, 255, 110), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx - 2, ty - 2), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Rotating geometric golden mandala rings framing forehead/head
        cx, cy = int(round(geom.face_center[0])), int(round(geom.face_center[1]))
        r_outer = max(24, int(round(geom.face_width * (0.85 + 0.06 * mystic_surge))))
        cv2.circle(canvas, (cx, cy), r_outer, accent_bgr, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), int(r_outer * 0.82), secondary_bgr, 1, cv2.LINE_AA)

        for offset in (0.0, math.pi / 4.0):
            sq = []
            r_sq = r_outer * 0.90
            for k in range(4):
                th = self._mandala_angle + offset + k * (math.pi / 2.0)
                sq.append([int(round(cx + math.cos(th) * r_sq)), int(round(cy + math.sin(th) * r_sq))])
            cv2.polylines(canvas, [np.array(sq, dtype=np.int32)], True, secondary_bgr, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 4. IRON MAN: Bold red & gold cel-shaded helmet + cyan eye slits
    # ------------------------------------------------------------------
    def _draw_comic_iron_man(self, canvas: np.ndarray, geom: FaceGeometry, ts: float) -> None:
        pts = geom.points
        r_vec, u_vec = geom.right_vec, geom.up_vec
        ink = (12, 12, 18)
        ink_w = max(2, int(round(geom.face_width * 0.013)))
        surge = float(np.clip(geom.mouth_open_ratio, 0.0, 1.5))

        # Crimson outer helmet shell
        helmet = canvas.copy()
        cv2.fillPoly(helmet, [geom.extended_oval], (24, 24, 192), cv2.LINE_AA)

        # Bold brushed gold faceplate polygon
        gold_plate = np.round(
            np.array(
                [
                    pts[103] + u_vec * (geom.face_height * 0.05),
                    pts[67] + u_vec * (geom.face_height * 0.06),
                    pts[9] + u_vec * (geom.face_height * 0.02),
                    pts[338] + u_vec * (geom.face_height * 0.06),
                    pts[297] + u_vec * (geom.face_height * 0.05),
                    pts[332],
                    pts[300],
                    pts[345],
                    pts[433],
                    pts[397],
                    pts[379],
                    pts[152],
                    pts[150],
                    pts[172],
                    pts[213],
                    pts[116],
                    pts[70],
                ],
                dtype=np.float32,
            )
        ).astype(np.int32)
        cv2.fillPoly(helmet, [gold_plate], (45, 198, 246), cv2.LINE_AA)
        cv2.addWeighted(helmet, 0.90, canvas, 0.10, 0.0, canvas)

        # Bold black comic armor seams & mouth slit
        cv2.polylines(canvas, [geom.extended_oval], True, ink, ink_w + 1, cv2.LINE_AA)
        cv2.polylines(canvas, [gold_plate], True, ink, ink_w, cv2.LINE_AA)
        brow_seam = np.round(np.array([pts[70], pts[105], pts[9], pts[334], pts[300]], dtype=np.float32)).astype(np.int32)
        cv2.polylines(canvas, [brow_seam], False, ink, ink_w, cv2.LINE_AA)

        mouth_slit = np.round(np.array([pts[61], pts[13], pts[291], pts[17]], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(canvas, [mouth_slit], (26, 26, 32), cv2.LINE_AA)
        cv2.polylines(canvas, [mouth_slit], True, ink, ink_w, cv2.LINE_AA)
        if surge > 0.20:
            mc = 0.5 * (pts[13] + pts[14])
            cv2.circle(canvas, (int(round(mc[0])), int(round(mc[1]))), max(3, int(round(geom.face_width * 0.03 * surge))), (255, 235, 90), -1, cv2.LINE_AA)

        # Glowing cyan comic eye slits + J.A.R.V.I.S. power-surge targeting arcs on mouth open
        eye_glow_bgr = (255, 248, 175) if surge > 0.20 else (255, 232, 65)
        spin_deg = (ts * 110.0) % 360.0
        for is_left in (True, False):
            if is_left:
                inner_c, outer_c = pts[133], pts[33]
                open_ratio, side = geom.left_eye_open, -1.0
            else:
                inner_c, outer_c = pts[362], pts[263]
                open_ratio, side = geom.right_eye_open, 1.0

            eye_c = 0.5 * (inner_c + outer_c)
            eye_w = max(float(np.linalg.norm(outer_c - inner_c)), geom.face_width * 0.14)
            slit_h = eye_w * (0.14 + 0.16 * open_ratio)
            slit = np.round(
                np.array(
                    [
                        eye_c - r_vec * (side * eye_w * 0.50) - u_vec * (slit_h * 0.35),
                        eye_c + r_vec * (side * eye_w * 0.72) + u_vec * (slit_h * 0.85),
                        eye_c + r_vec * (side * eye_w * 0.82) + u_vec * (slit_h * 0.15),
                        eye_c - r_vec * (side * eye_w * 0.42) - u_vec * (slit_h * 0.85),
                    ],
                    dtype=np.float32,
                )
            ).astype(np.int32)
            if surge > 0.20:
                cv2.polylines(canvas, [slit], True, (255, 235, 90), ink_w + 4, cv2.LINE_AA)
                ecx, ecy = int(round(eye_c[0])), int(round(eye_c[1]))
                arc_r = max(8, int(round(eye_w * (0.72 + 0.22 * surge))))
                cv2.ellipse(canvas, (ecx, ecy), (arc_r, arc_r), geom.roll_deg, spin_deg, spin_deg + 85.0, (255, 245, 130), 2, cv2.LINE_AA)
                cv2.ellipse(canvas, (ecx, ecy), (arc_r, arc_r), geom.roll_deg, spin_deg + 180.0, spin_deg + 265.0, (255, 245, 130), 2, cv2.LINE_AA)
            cv2.fillPoly(canvas, [slit], eye_glow_bgr, cv2.LINE_AA)
            cv2.polylines(canvas, [slit], True, ink, ink_w + 1, cv2.LINE_AA)

    def _draw_extra_hero(self, canvas: np.ndarray, geom: FaceGeometry, slug: str, ts: float) -> None:
        """Landmark-anchored comic costume masks for the expanded hero roster."""
        palette = {
            "thor": ((105, 115, 155), (240, 225, 180)),
            "loki": ((35, 95, 45), (45, 210, 220)),
            "black_panther": ((32, 25, 35), (175, 95, 190)),
            "deadpool": ((35, 35, 205), (22, 22, 30)),
            "captain_america": ((155, 65, 35), (245, 245, 245)),
            "wolverine": ((35, 205, 245), (180, 75, 25)),
            "venom": ((19, 19, 24), (250, 250, 250)),
            "ghost_rider": ((215, 215, 225), (0, 155, 255)),
        }
        base, accent = palette[slug]
        oval = geom.extended_oval.astype(np.int32)
        w = max(18.0, float(geom.face_width))
        h = max(25.0, float(geom.face_height))
        center = geom.face_center
        right, up = geom.right_vec, geom.up_vec
        ink = (12, 12, 19)
        def pt(x: float, y: float) -> tuple[int, int]:
            pos = center + right * (x * w) + up * (y * h)
            return int(round(pos[0])), int(round(pos[1]))
        def polygon(coords, color, outline=True):
            points = np.array([pt(x, y) for x, y in coords], dtype=np.int32)
            cv2.fillPoly(canvas, [points], color, lineType=cv2.LINE_AA)
            if outline:
                cv2.polylines(canvas, [points], True, ink, max(2, int(w * .014)), cv2.LINE_AA)
        cv2.fillPoly(canvas, [oval], base, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [oval], True, ink, max(2, int(w * .018)), cv2.LINE_AA)
        if slug == "loki":
            polygon([(-.38,.30),(-.63,.95),(-.40,.78),(-.22,.47),(.22,.47),(.40,.78),(.63,.95),(.38,.30)],accent)
            polygon([(-.30,.28),(.30,.28),(.17,.07),(-.17,.07)],(40,135,65))
        elif slug == "thor":
            for side in (-1,1):
                polygon([(side*.28,.29),(side*.61,.48),(side*.63,.73),(side*.35,.55)],(210,215,225))
            polygon([(-.28,.38),(.28,.38),(.35,.19),(-.35,.19)],(120,120,155))
        elif slug == "black_panther":
            polygon([(-.43,.29),(-.56,.67),(-.17,.44)],base)
            polygon([(.43,.29),(.56,.67),(.17,.44)],base)
            for side in (-1,1):
                polygon([(side*.09,-.30),(side*.40,-.17),(side*.30,-.04),(side*.10,-.13)],accent)
            polygon([(-.10,-.47),(0,-.58),(.10,-.47),(0,-.34)],accent)
        elif slug == "deadpool":
            for side in (-1,1):
                polygon([(side*.05,.15),(side*.40,.24),(side*.34,-.18),(side*.10,-.12)],(24,24,28))
                polygon([(side*.11,.06),(side*.32,.08),(side*.26,-.05),(side*.12,-.04)],(245,245,245))
            polygon([(-.10,-.35),(.10,-.35),(.07,-.46),(-.07,-.46)],(25,25,30))
        elif slug == "captain_america":
            polygon([(-.40,.15),(-.40,.53),(.40,.53),(.40,.15)],(155,65,35))
            polygon([(-.16,.24),(0,.52),(.16,.24),(.09,.24),(.04,.33),(-.04,.33),(-.09,.24)],accent)
            for side in (-1,1):
                polygon([(side*.09,-.08),(side*.40,.05),(side*.34,-.12),(side*.12,-.16)],(245,245,245))
        elif slug == "wolverine":
            for side in (-1,1):
                polygon([(side*.12,.20),(side*.53,.80),(side*.48,.20),(side*.37,-.12)],accent)
                polygon([(side*.09,.08),(side*.36,.05),(side*.30,-.13),(side*.10,-.12)],(240,245,250))
            polygon([(-.12,-.28),(0,-.43),(.12,-.28)],accent)
        elif slug == "venom":
            for side in (-1,1):
                polygon([(side*.03,.08),(side*.39,.23),(side*.46,-.01),(side*.13,-.16)],(250,250,250))
            for i in range(7):
                x=-.32+i*.105
                polygon([(x,-.31),(x+.05,-.31),(x+.025,-.46)],(245,245,245),False)
            polygon([(-.35,-.27),(.35,-.27),(.27,-.36),(-.27,-.36)],(20,30,165))
        elif slug == "ghost_rider":
            # Comic skull: eye sockets, nasal cavity, teeth and animated flames.
            for side in (-1,1):
                polygon([(side*.06,.13),(side*.34,.20),(side*.29,-.08),(side*.10,-.06)],(15,15,20))
            polygon([(-.07,-.18),(0,-.30),(.07,-.18)],(15,15,20))
            for i in range(6):
                x=-.27+i*.10
                cv2.line(canvas,pt(x,-.36),pt(x,-.48),ink,max(1,int(w*.012)),cv2.LINE_AA)
            phase = .04 * np.sin(ts*5.0)
            for i in range(7):
                x=-.44+i*.145
                polygon([(x-.10,.35),(x+.10,.35),(x+.04,.70+phase+(.12 if i%2 else 0))],(0,95+int(i%3)*35,255),False)
        if slug in {"thor", "loki"}:
            for side in (-1,1):
                polygon([(side*.08,.06),(side*.34,.08),(side*.30,-.05),(side*.10,-.04)],(235,240,245))
