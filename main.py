"""Marvel Portal & Gesture Controller — Cross-platform application entry point.

Tracks both hands in real time via webcam using MediaPipe Hands, renders a
4-point perspective-warped Marvel character portal between the two hands
(with single-hand HUD fallback), and switches characters with zero false
positives when Thumb Tip (4) and Middle Finger Tip (12) tap together.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import platform
import queue
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple
import urllib.request

import cv2
import numpy as np

from gesture_engine import HandTelemetry, ZeroErrorPinchDetector
from portal_renderer import MarvelPortalRenderer, PipelineTimings


WINDOW_NAME = "Marvel Portal Controller"
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_VIRTUAL_CAMERA_KEYWORDS = (
    "iriun",
    "virtual",
    "obs",
    "droidcam",
    "epoccam",
    "camo",
    "manycam",
    "xsplit",
)


@dataclass
class _DiscoveredCamera:
    """Verified working camera candidate returned during startup scanning."""

    hw_index: int
    name: str
    badge: str
    width: int
    height: int
    cap: cv2.VideoCapture
    is_virtual: bool


def _get_windows_camera_names() -> List[str]:
    """Return DirectShow camera names on Windows via pygrabber, or [] on fallback."""
    if platform.system() != "Windows":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception:
        return []


def _is_virtual_camera_name(name: str) -> bool:
    """Heuristically identify software/virtual camera devices by name."""
    lower = name.lower()
    return any(keyword in lower for keyword in _VIRTUAL_CAMERA_KEYWORDS)


def _candidate_backends(system_name: str) -> List[Tuple[int, str]]:
    """Return ordered (OpenCV backend ID, HUD badge) candidates for the current OS."""
    if system_name == "Windows":
        return [
            (cv2.CAP_DSHOW, "WIN-DSHOW"),
            (cv2.CAP_MSMF, "WIN-MSMF"),
            (cv2.CAP_ANY, "WIN-ANY"),
        ]
    if system_name == "Darwin":
        return [
            (cv2.CAP_AVFOUNDATION, "MAC-AVF"),
            (cv2.CAP_ANY, "MAC-ANY"),
        ]
    default_badge = "LINUX-V4L2" if system_name == "Linux" else f"{system_name.upper()[:6]}-CAM"
    return [(cv2.CAP_ANY, default_badge)]


def _probe_single_index(
    idx: int,
    backends: Sequence[Tuple[int, str]],
) -> Optional[Tuple[cv2.VideoCapture, int, str, int, int]]:
    """Probe one camera index across OS backends and verify a valid frame read."""
    for backend, badge in backends:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        # Keep internal frame queue minimal when supported by the backend
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(2):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                actual_h, actual_w = frame.shape[:2]
                return cap, backend, badge, int(actual_w), int(actual_h)

        cap.release()
    return None


def create_camera_capture(camera_index: int = 0) -> Tuple[cv2.VideoCapture, str]:
    """Open the webcam using the lowest-latency OS-native backend.

    Tries `camera_index` first, then automatically scans indices 0..5 on Windows
    (0..4 on macOS/Linux) across OS-specific backends and verifies `cap.read()`.

    Returns:
        (VideoCapture instance, short HUD backend badge string)
    """
    system_name = platform.system()
    backends = _candidate_backends(system_name)
    scan_end = 6 if system_name == "Windows" else 5
    candidate_indices = [camera_index] + [idx for idx in range(scan_end) if idx != camera_index]

    for idx in candidate_indices:
        probed = _probe_single_index(idx, backends)
        if probed is not None:
            cap, _, badge, _, _ = probed
            if idx != camera_index:
                print(f"[CAM] Requested index {camera_index} unavailable; selected index {idx} ({badge}).")
            return cap, badge

    fallback_badge = backends[-1][1]
    return cv2.VideoCapture(), fallback_badge


def discover_and_select_camera(
    cli_camera_index: Optional[int] = None,
) -> Tuple[cv2.VideoCapture, str]:
    """Scan available cameras and interactively prompt selection unless CLI index is set."""
    if cli_camera_index is not None:
        return create_camera_capture(camera_index=cli_camera_index)

    system_name = platform.system()
    backends = _candidate_backends(system_name)
    scan_range = range(6) if system_name == "Windows" else range(5)
    device_names = _get_windows_camera_names()

    print("[CAM] Scanning available cameras...")
    discovered: List[_DiscoveredCamera] = []

    for idx in scan_range:
        probed = _probe_single_index(idx, backends)
        name = device_names[idx] if idx < len(device_names) else f"Camera {idx}"
        if probed is None:
            if idx < len(device_names):
                print(f"[CAM] Skipped unavailable/disabled device #{idx}: {name}")
            continue

        cap, _, badge, actual_w, actual_h = probed
        discovered.append(
            _DiscoveredCamera(
                hw_index=idx,
                name=name,
                badge=badge,
                width=actual_w,
                height=actual_h,
                cap=cap,
                is_virtual=_is_virtual_camera_name(name),
            )
        )

    if not discovered:
        return cv2.VideoCapture(), backends[-1][1]

    # Prefer physical cameras first so [0] defaults to the primary physical webcam
    discovered.sort(key=lambda cam: (cam.is_virtual, cam.hw_index))

    print("\nAvailable Cameras:")
    for menu_idx, cam in enumerate(discovered):
        print(f"  [{menu_idx}] {cam.name} ({cam.badge} - {cam.width}x{cam.height})")

    if len(discovered) == 1:
        selected = discovered[0]
        print(f"Auto-selected camera [0]: {selected.name} ({selected.badge} - {selected.width}x{selected.height})\n")
        return selected.cap, selected.badge

    selected_idx = 0
    while True:
        try:
            raw_choice = input("Select camera number [0]: ").strip()
        except EOFError:
            selected_idx = 0
            break

        if raw_choice == "":
            selected_idx = 0
            break
        if raw_choice.isdigit():
            parsed = int(raw_choice)
            if 0 <= parsed < len(discovered):
                selected_idx = parsed
                break
        print(f"Please enter a number between 0 and {len(discovered) - 1}.")

    selected = discovered[selected_idx]
    for idx, cam in enumerate(discovered):
        if idx != selected_idx:
            cam.cap.release()

    print(f"Selected [{selected_idx}]: {selected.name} ({selected.badge} - {selected.width}x{selected.height})\n")
    return selected.cap, selected.badge


def _make_synthetic_landmarks(
    wrist: Tuple[float, float],
    middle_mcp: Tuple[float, float],
    thumb_tip: Tuple[float, float],
    index_tip: Tuple[float, float],
    middle_tip: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Construct a 21-point normalized landmark list for headless self-testing."""
    pts = [wrist for _ in range(21)]
    pts[0] = wrist
    pts[4] = thumb_tip
    pts[8] = index_tip
    pts[9] = middle_mcp
    pts[12] = middle_tip
    return pts


def run_self_test() -> int:
    """Execute all 12 mandatory acceptance tests across tracking, geometry, gestures, and rendering."""
    print("[SELF-TEST] Running 12 Mandatory Acceptance Tests...")
    renderer = MarvelPortalRenderer()
    geom_engine = renderer.geometry_engine
    expected_slugs = ["spider_man", "scarlet_witch", "doctor_strange", "iron_man", "thor", "loki", "black_panther", "deadpool", "captain_america", "wolverine", "venom", "ghost_rider"]
    assert [c.slug for c in renderer.characters] == expected_slugs

    detector = ZeroErrorPinchDetector(
        trigger_threshold=0.29,
        release_threshold=0.38,
        ema_alpha=0.82,
        required_consecutive_frames=1,
        cooldown_seconds=0.25,
        index_extension_factor=0.85,
        index_cancel_factor=0.78,
    )

    w, h = 1280, 720

    def _build_hand_21(
        wrist: Tuple[float, float] = (0.25, 0.70),
        idx_mcp: Tuple[float, float] = (0.21, 0.52),
        mid_mcp: Tuple[float, float] = (0.25, 0.50),
        pnk_mcp: Tuple[float, float] = (0.31, 0.54),
        thm4: Tuple[float, float] = (0.20, 0.50),
        idx8: Tuple[float, float] = (0.18, 0.20),
        mid12: Tuple[float, float] = (0.25, 0.16),
        rng16: Tuple[float, float] = (0.29, 0.18),
        pnk20: Tuple[float, float] = (0.33, 0.22),
    ) -> List[Tuple[float, float]]:
        pts = [wrist for _ in range(21)]
        pts[0] = wrist
        pts[4] = thm4
        pts[5] = idx_mcp
        pts[8] = idx8
        pts[9] = mid_mcp
        pts[12] = mid12
        pts[16] = rng16
        pts[17] = pnk_mcp
        pts[20] = pnk20
        return pts

    # TEST 1: Four-Point Control (exact tracking of Left/Right Thumb 4 & Index 8 regardless of order)
    h_left = _build_hand_21(thm4=(0.20, 0.55), idx8=(0.20, 0.20))
    h_right = _build_hand_21(thm4=(0.80, 0.55), idx8=(0.80, 0.20))
    g_normal = geom_engine.compute_geometry([(h_left, "Left"), (h_right, "Right")], w, h)
    g_swapped = geom_engine.compute_geometry([(h_right, "Right"), (h_left, "Left")], w, h)
    assert g_normal.is_valid and g_swapped.is_valid and g_normal.quad is not None and g_swapped.quad is not None
    assert np.array_equal(g_normal.quad, g_swapped.quad), "Handedness resolution must ignore detection array order!"
    assert g_normal.p1_left_thumb == (int(round(0.20 * w)), int(round(0.55 * h)))
    assert g_normal.p2_left_index == (int(round(0.20 * w)), int(round(0.20 * h)))
    assert g_normal.p3_right_index == (int(round(0.80 * w)), int(round(0.20 * h)))
    assert g_normal.p4_right_thumb == (int(round(0.80 * w)), int(round(0.55 * h)))
    print("[PASS] TEST 1: Four-Point Control & Handedness Resolution")

    # TEST 2: Other Fingers (12, 16, 20, wrist, palm) must have ZERO effect on portal geometry
    h_l_curl = _build_hand_21(thm4=(0.20, 0.55), idx8=(0.20, 0.20), mid12=(0.22, 0.72), rng16=(0.23, 0.74), pnk20=(0.25, 0.76), wrist=(0.12, 0.85))
    h_r_curl = _build_hand_21(thm4=(0.80, 0.55), idx8=(0.80, 0.20), mid12=(0.78, 0.72), rng16=(0.77, 0.74), pnk20=(0.75, 0.76), wrist=(0.90, 0.85))
    g_curl = geom_engine.compute_geometry([(h_l_curl, "Left"), (h_r_curl, "Right")], w, h)
    assert g_curl.is_valid and np.array_equal(g_normal.quad, g_curl.quad), "Other fingers affected portal quad!"
    print("[PASS] TEST 2: Other Fingers Zero-Influence Isolation")

    # TEST 3: Maximum Hand Separation (no artificial size cap)
    h_l_edge = _build_hand_21(thm4=(0.01, 0.99), idx8=(0.01, 0.01))
    h_r_edge = _build_hand_21(thm4=(0.99, 0.99), idx8=(0.99, 0.01))
    g_wide = geom_engine.compute_geometry([(h_l_edge, "Left"), (h_r_edge, "Right")], w, h)
    assert g_wide.is_valid and (g_wide.bbox[2] - g_wide.bbox[0]) > 1240 and (g_wide.bbox[3] - g_wide.bbox[1]) > 690
    print("[PASS] TEST 3: Maximum Hand Separation (Unclamped Expansion)")

    # TEST 4: Hand Approach (proportional contraction without fixed minimum dimension distortion)
    h_l_close = _build_hand_21(thm4=(0.48, 0.52), idx8=(0.48, 0.48))
    h_r_close = _build_hand_21(thm4=(0.52, 0.52), idx8=(0.52, 0.48))
    g_close = geom_engine.compute_geometry([(h_l_close, "Left"), (h_r_close, "Right")], w, h)
    assert g_close.is_valid and g_close.area < 2000.0 and g_wide.area > g_close.area * 200.0
    print("[PASS] TEST 4: Proportional Hand Contraction")

    # TEST 5: Rapid Movement (1:1 immediate corner tracking, zero EMA/grace delay)
    t0 = 10.0
    open_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.43, 0.42), (0.48, 0.24), (0.54, 0.35))
    moved_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.61, 0.29), (0.57, 0.18), (0.69, 0.22))
    detector.reset_missing_hands(set())
    detector.process_hand(open_lms, "Right", w, h, now=t0)
    tel_moved = detector.process_hand(moved_lms, "Right", w, h, now=t0 + 0.016)
    assert tel_moved.thumb_tip_px == (int(round(0.61 * w)), int(round(0.29 * h)))
    assert tel_moved.index_tip_px == (int(round(0.57 * w)), int(round(0.18 * h)))
    g_jump = geom_engine.compute_geometry([(h_l_edge, "Left"), (h_r_close, "Right")], w, h)
    assert g_jump.p1_left_thumb == (int(round(0.01 * w)), int(round(0.99 * h)))
    assert g_jump.p3_right_index == (int(round(0.52 * w)), int(round(0.48 * h)))
    print("[PASS] TEST 5: Rapid Movement Zero-Lag Response")

    # TEST 6: Hand Rotation + Self-Intersection Repair + Identical Handedness Disambiguation + Missing Hand Safety
    g_missing = geom_engine.compute_geometry([(h_left, "Left")], w, h)
    assert not g_missing.is_valid and g_missing.rejection_reason == "MISSING_HAND"
    g_same_lbl = geom_engine.compute_geometry([(h_right, "Left"), (h_left, "Left")], w, h)
    assert g_same_lbl.is_valid and np.array_equal(g_same_lbl.quad, g_normal.quad)
    # Crossed/rotated quad that would self-intersect under naive ordering must untwist cleanly
    untwisted = geom_engine.order_and_untwist_quad((100, 100), (300, 110), (140, 260), (260, 140))
    pts_u = [tuple(map(int, p)) for p in untwisted]
    assert not _segments_intersect_helper(pts_u[0], pts_u[1], pts_u[2], pts_u[3])
    assert not _segments_intersect_helper(pts_u[1], pts_u[2], pts_u[3], pts_u[0])
    # Rotated/foreshortened hand scale invariance (Wrist 0->9 foreshortens, MCP 5->17 preserves scale)
    rotated_pinch = _build_hand_21(
        wrist=(0.50, 0.52),
        idx_mcp=(0.45, 0.50),
        mid_mcp=(0.50, 0.50),   # foreshortened longitudinal only 0.02*h!
        pnk_mcp=(0.57, 0.50),   # transverse 0.12*w preserves true palm scale!
        thm4=(0.50, 0.42),
        idx8=(0.48, 0.30),
        mid12=(0.505, 0.424),
    )
    detector.reset_missing_hands(set())
    tel_rot = detector.process_hand(rotated_pinch, "Left", w, h, now=t0 + 1.0)
    assert tel_rot.palm_ref_length > 100.0 and tel_rot.state == "PINCH_HELD"
    print("[PASS] TEST 6: Hand Rotation, Un-Twisted Quad & Missing-Hand Safety")

    # TEST 7: Frame Boundaries (out-of-frame coords stay undistorted in quad, ROI safely clipped)
    h_l_oob = _build_hand_21(thm4=(-0.05, 1.04), idx8=(-0.03, 0.15))
    h_r_oob = _build_hand_21(thm4=(1.05, 0.85), idx8=(1.02, -0.04))
    g_oob = geom_engine.compute_geometry([(h_l_oob, "Left"), (h_r_oob, "Right")], w, h)
    assert g_oob.is_valid and int(np.min(g_oob.quad[:, 0])) < 0 and g_oob.bbox == (0, 0, w, h)
    oob_frame = np.full((h, w, 3), 120, dtype=np.uint8)
    oob_out, oob_mode = renderer.render_ar_portal_scene(oob_frame, [(h_l_oob, "Left"), (h_r_oob, "Right")])
    assert oob_mode == "HANDS: 2/2 [PORTAL ACTIVE]"
    print("[PASS] TEST 7: Screen Boundary Tracking & Unclamped Vertices")

    # TEST 8: Character Selection (Thumb 4 + Middle 12 Release-Edge + Anti-Fist + Multi-hand Independence)
    fist_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.50, 0.30), (0.50, 0.62), (0.505, 0.305))
    ready_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.478, 0.37), (0.48, 0.52), (0.522, 0.37))
    pinch_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.50, 0.36), (0.48, 0.52), (0.504, 0.364))
    mid_lms = _make_synthetic_landmarks((0.50, 0.70), (0.50, 0.50), (0.482, 0.37), (0.48, 0.52), (0.519, 0.37))

    detector.reset_missing_hands(set())
    detector.process_hand(ready_lms, "Right", w, h, now=t0 + 2.00)
    assert detector.process_hand(pinch_lms, "Right", w, h, now=t0 + 2.03).state == "PINCH_HELD"
    assert not detector.process_hand(mid_lms, "Right", w, h, now=t0 + 2.06).triggered_this_frame
    assert detector.process_hand(open_lms, "Right", w, h, now=t0 + 2.30).triggered_this_frame
    assert not detector.process_hand(fist_lms, "Right", w, h, now=t0 + 2.60).triggered_this_frame
    print("[PASS] TEST 8: Release-Edge Character Selection & Anti-Fist Guard")

    # TEST 9: Comic Effect Isolation across all 4 heroes (0 changes outside quad)
    base_frame = np.full((h, w, 3), 160, dtype=np.uint8)
    cv2.rectangle(base_frame, (440, 180), (840, 540), (70, 120, 190), -1)
    h_box_l = _build_hand_21(thm4=(390 / w, 560 / h), idx8=(390 / w, 160 / h))
    h_box_r = _build_hand_21(thm4=(890 / w, 560 / h), idx8=(890 / w, 160 / h))

    for _ in renderer.characters:
        zero_out, zero_mode = renderer.render_ar_portal_scene(base_frame, [])
        assert zero_mode == "HANDS: 0/2" and np.array_equal(zero_out, base_frame)
        single_out, single_mode = renderer.render_ar_portal_scene(base_frame, [(h_box_r, "Right")])
        assert single_mode == "HANDS: 1/2" and np.array_equal(single_out, base_frame)

        dual_out, dual_mode = renderer.render_ar_portal_scene(base_frame, [(h_box_l, "Left"), (h_box_r, "Right")])
        assert dual_mode == "HANDS: 2/2 [PORTAL ACTIVE]"
        assert not np.array_equal(dual_out[200:520, 480:800], base_frame[200:520, 480:800])
        assert np.array_equal(dual_out[0:140, :], base_frame[0:140, :])
        assert np.array_equal(dual_out[:, 0:370], base_frame[:, 0:370])
        assert np.array_equal(dual_out[:, 910:w], base_frame[:, 910:w])
        renderer.switch_character(step=1, source_label="SelfTest")
    renderer.reset_character()
    print("[PASS] TEST 9: Strict Portal Comic Effect Isolation (All 4 Heroes)")

    # TEST 10: Real-Time Performance Budget & HandLandmarker check
    with _UnifiedHandTracker(renderer.assets_dir, max_num_hands=2, min_detection_confidence=0.30, min_presence_confidence=0.30, min_tracking_confidence=0.30) as hand_tr:
        test_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        hand_res = hand_tr.process(test_rgb)
        assert isinstance(hand_res, list)
    t_r0 = time.perf_counter()
    for _ in range(10):
        renderer.render_ar_portal_scene(base_frame.copy(), [(h_box_l, "Left"), (h_box_r, "Right")], in_place=True)
    avg_render_ms = (time.perf_counter() - t_r0) * 1000.0 / 10.0
    assert avg_render_ms < 12.0, f"Optimized ROI portal render exceeded budget: {avg_render_ms:.2f} ms"
    print(f"[PASS] TEST 10: ROI Shader + Precomputed LUT Budget ({avg_render_ms:.2f} ms/frame)")

    # TEST 11: Model Failure Isolation (Optional FaceTracker failure cannot affect portal)
    opt_face = _OptionalFaceTracker(renderer.assets_dir, inference_interval=3)
    opt_face.simulate_failure = True
    failed_faces = opt_face.process_roi(np.zeros((h, w, 3), dtype=np.uint8), (390, 160, 890, 560), w, h)
    assert failed_faces == [] and opt_face.failure_count == 1
    safe_out, safe_mode = renderer.render_ar_portal_scene(
        base_frame, [(h_box_l, "Left"), (h_box_r, "Right")], face_landmarks_list=failed_faces
    )
    assert safe_mode == "HANDS: 2/2 [PORTAL ACTIVE]"
    print("[PASS] TEST 11: Optional Model Failure Isolation")

    # TEST 12: Bounded Async Recorder Queue & Stability (Zero Unbounded Growth)
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_caps = Path(tmp_dir_str) / "captures"
        snap_path = save_clean_snapshot(safe_out, captures_dir=tmp_caps)
        assert snap_path.is_file() and snap_path.stat().st_size > 1000

        rec = _CleanVideoRecorder(captures_dir=tmp_caps, fps=30.0, max_queue_size=4)
        rec.toggle(w, h)
        assert rec.is_recording
        work_buf = base_frame.copy()
        for idx_f in range(120):
            renderer.render_ar_portal_scene(work_buf, [(h_box_l, "Left"), (h_box_r, "Right")], in_place=True)
            rec.write(work_buf)
        assert rec.queue_size <= 4
        mp4_path = rec.stop()
        assert mp4_path is not None and mp4_path.is_file() and mp4_path.stat().st_size > 0 and not rec.is_recording
    print("[PASS] TEST 12: Bounded Async Recorder & Extended Stability")

    # Mandatory Expansion Validation: TEST A (Close) -> TEST B (+25%) -> TEST C (+50%) -> TEST D (Edges)
    expansion_cases = [
        ("TEST A (Close)", 0.44, 0.56),
        ("TEST B (+25%)", 0.425, 0.575),
        ("TEST C (+50%)", 0.41, 0.59),
        ("TEST D (Edges)", 0.03, 0.97),
    ]
    prev_ren_w = 0
    for label, lx, rx in expansion_cases:
        lh_case = _build_hand_21(thm4=(lx, 0.58), idx8=(lx, 0.32))
        rh_case = _build_hand_21(thm4=(rx, 0.58), idx8=(rx, 0.32))
        geom_case = geom_engine.compute_geometry([(lh_case, "Left"), (rh_case, "Right")], w, h)
        t_case = PipelineTimings()
        out_case, mode_case = renderer.render_ar_portal_scene(
            base_frame, [(lh_case, "Left"), (rh_case, "Right")], portal_geometry=geom_case, timings=t_case
        )
        assert mode_case == "HANDS: 2/2 [PORTAL ACTIVE]"
        diff_mask = np.any(out_case != base_frame, axis=2)
        ys, xs = np.where(diff_mask)
        ren_w = int(xs.max() - xs.min() + 1) if len(xs) else 0
        assert ren_w > prev_ren_w, f"{label}: rendered portal width ({ren_w}) did not exceed {prev_ren_w}"
        prev_ren_w = ren_w
        print(
            f"[PASS] {label}: geom_bbox=({geom_case.bbox[2]-geom_case.bbox[0]}x{geom_case.bbox[3]-geom_case.bbox[1]}) "
            f"| rendered_diff_w={ren_w}px | dest_wh={t_case.dest_wh}"
        )

    return 0


def _segments_intersect_helper(
    a: Tuple[int, int],
    b: Tuple[int, int],
    c: Tuple[int, int],
    d: Tuple[int, int],
) -> bool:
    """Self-test geometric crossing predicate."""
    def _orient(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
        val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 1 if val > 0 else (-1 if val < 0 else 0)

    return (_orient(a, b, c) * _orient(a, b, d) < 0) and (_orient(c, d, a) * _orient(c, d, b) < 0)


def _ensure_model_file(assets_dir: Path, filename: str, url: str) -> Path:
    """Ensure a MediaPipe `.task` model file exists in `assets/`, downloading if needed."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    model_path = assets_dir / filename
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path

    tmp_path = model_path.with_suffix(".task.tmp")
    print(f"[INIT] Downloading MediaPipe {filename} to {model_path}...")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.replace(model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return model_path


def ensure_hand_landmarker_model(assets_dir: Path) -> Path:
    """Ensure the official MediaPipe HandLandmarker task file exists in assets/."""
    return _ensure_model_file(assets_dir, "hand_landmarker.task", HAND_LANDMARKER_MODEL_URL)


def ensure_face_landmarker_model(assets_dir: Path) -> Path:
    """Ensure the official MediaPipe FaceLandmarker task file exists in assets/."""
    return _ensure_model_file(assets_dir, "face_landmarker.task", FACE_LANDMARKER_MODEL_URL)


class _OptionalFaceTracker:
    """Optional, throttled ROI FaceLandmarker isolated from hand tracking.

    Runs only on cropped portal ROIs every Nth frame (`inference_interval=3`),
    caches remapped normalized coordinates, and isolates all errors so a face
    model failure never degrades hand tracking or portal geometry.
    """

    def __init__(self, assets_dir: Path, inference_interval: int = 3) -> None:
        self.assets_dir = assets_dir
        self.inference_interval = max(1, int(inference_interval))
        self.simulate_failure: bool = False
        self.failure_count: int = 0
        self.last_error: str = ""

        self._mp = None
        self._tasks_landmarker = None
        self._initialized = False
        self._frame_counter = 0
        self._last_video_timestamp_ms = -1
        self._cached_faces: List[List[Tuple[float, float]]] = []

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._tasks_landmarker is not None
        self._initialized = True
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            self._mp = mp
            model_path = ensure_face_landmarker_model(self.assets_dir)
            try:
                base_options = BaseOptions(model_asset_buffer=model_path.read_bytes())
            except Exception:
                base_options = BaseOptions(model_asset_path=str(model_path))

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_facial_transformation_matrixes=False,
            )
            self._tasks_landmarker = vision.FaceLandmarker.create_from_options(options)
            return True
        except Exception as exc:
            self.failure_count += 1
            self.last_error = str(exc)
            self._tasks_landmarker = None
            return False

    def close(self) -> None:
        """Release underlying MediaPipe resources safely."""
        if self._tasks_landmarker is not None:
            self._tasks_landmarker.close()
            self._tasks_landmarker = None

    def process_roi(
        self,
        rgb_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        full_w: int,
        full_h: int,
    ) -> List[List[Tuple[float, float]]]:
        """Run throttled FaceLandmarker on `bbox` ROI and return cached full-frame coords."""
        try:
            if self.simulate_failure:
                raise RuntimeError("Simulated optional face detection failure")

            x1, y1, x2, y2 = bbox
            rw, rh = x2 - x1, y2 - y1
            if rw <= 15 or rh <= 15:
                return []

            should_infer = (self._frame_counter % self.inference_interval == 0)
            self._frame_counter += 1
            if not should_infer:
                return self._cached_faces

            if not self._ensure_initialized() or self._tasks_landmarker is None or self._mp is None:
                return []

            # Face detection needs context around the portal; a tight hand-defined
            # rectangle frequently clips the forehead/chin and prevents detection.
            pad_x = max(80, int(rw * 0.40))
            pad_y = max(80, int(rh * 0.55))
            crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            crop_x2, crop_y2 = min(full_w, x2 + pad_x), min(full_h, y2 + pad_y)
            crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
            roi_rgb = np.ascontiguousarray(rgb_frame[crop_y1:crop_y2, crop_x1:crop_x2])
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=roi_rgb)
            timestamp_ms = max(int(time.perf_counter() * 1000), self._last_video_timestamp_ms + 1)
            self._last_video_timestamp_ms = timestamp_ms
            results = self._tasks_landmarker.detect_for_video(mp_image, timestamp_ms)

            remapped: List[List[Tuple[float, float]]] = []
            if results.face_landmarks:
                inv_w = 1.0 / max(1.0, float(full_w))
                inv_h = 1.0 / max(1.0, float(full_h))
                for face_lms in results.face_landmarks:
                    pts = [
                        ((crop_x1 + float(lm.x) * crop_w) * inv_w, (crop_y1 + float(lm.y) * crop_h) * inv_h)
                        for lm in face_lms
                    ]
                    remapped.append(pts)
            self._cached_faces = remapped
            return self._cached_faces
        except Exception as exc:
            self.failure_count += 1
            self.last_error = str(exc)
            if self.failure_count <= 3:
                print(f"[FACE ERROR] {type(exc).__name__}: {exc}", flush=True)
            self._cached_faces = []
            return []


class _UnifiedHandTracker:
    """Stateless MediaPipe `HandLandmarker` (`RunningMode.IMAGE`) and solutions wrapper."""

    def __init__(
        self,
        assets_dir: Path,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.30,
        min_presence_confidence: float = 0.30,
        min_tracking_confidence: float = 0.30,
    ) -> None:
        self.assets_dir = assets_dir
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._mp = None
        self._backend_name = ""
        self._solutions_hands = None
        self._tasks_landmarker = None
        self._last_video_timestamp_ms = -1

    def __enter__(self) -> "_UnifiedHandTracker":
        import mediapipe as mp

        self._mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._backend_name = "solutions"
            self._solutions_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return self

        from mediapipe.tasks.python import BaseOptions, vision

        self._backend_name = "tasks"
        model_path = ensure_hand_landmarker_model(self.assets_dir)

        try:
            base_options = BaseOptions(model_asset_buffer=model_path.read_bytes())
        except Exception:
            base_options = BaseOptions(model_asset_path=str(model_path))

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._tasks_landmarker = vision.HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._solutions_hands is not None:
            self._solutions_hands.close()
            self._solutions_hands = None
        if self._tasks_landmarker is not None:
            self._tasks_landmarker.close()
            self._tasks_landmarker = None

    def process(self, rgb_frame: np.ndarray) -> List[Tuple[object, str]]:
        """Detect hands on the current frame and return raw `[(landmarks, label), ...]` with no caching."""
        raw_hands: List[Tuple[object, str]] = []

        if self._backend_name == "solutions" and self._solutions_hands is not None:
            was_writeable = rgb_frame.flags.writeable
            try:
                rgb_frame.flags.writeable = False
                results = self._solutions_hands.process(rgb_frame)
            finally:
                rgb_frame.flags.writeable = was_writeable
            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness or []
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    if idx < len(handedness_list) and handedness_list[idx].classification:
                        hand_label = handedness_list[idx].classification[0].label
                    else:
                        hand_label = f"Hand{idx}"
                    raw_hands.append((hand_lms, hand_label))
            return raw_hands

        if self._backend_name == "tasks" and self._tasks_landmarker is not None and self._mp is not None:
            contiguous_rgb = rgb_frame if rgb_frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(rgb_frame)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=contiguous_rgb)
            timestamp_ms = max(int(time.perf_counter() * 1000), self._last_video_timestamp_ms + 1)
            self._last_video_timestamp_ms = timestamp_ms
            results = self._tasks_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                frame_h, frame_w = rgb_frame.shape[:2]
                handedness_list = results.handedness or []
                scored: List[Tuple[object, str, float, float, float]] = []
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    if idx < len(handedness_list) and handedness_list[idx]:
                        cat = handedness_list[idx][0]
                        hand_label = cat.category_name or cat.display_name or f"Hand{idx}"
                        score = float(cat.score) if cat.score is not None else 0.5
                    else:
                        hand_label = f"Hand{idx}"
                        score = 0.5
                    wx = float(hand_lms[0].x) * frame_w
                    wy = float(hand_lms[0].y) * frame_h
                    scored.append((hand_lms, hand_label, score, wx, wy))

                # Stateless same-frame duplicate-hand filter: prevents 1 hand being detected twice
                # and collapsing the 2-hand portal onto a single hand.
                scored.sort(key=lambda item: item[2], reverse=True)
                accepted_wrists: List[Tuple[float, float]] = []
                for hand_lms, hand_label, _score, wx, wy in scored:
                    if any(((wx - ax) ** 2 + (wy - ay) ** 2) < (35.0 ** 2) for ax, ay in accepted_wrists):
                        continue
                    accepted_wrists.append((wx, wy))
                    raw_hands.append((hand_lms, hand_label))
                    if len(raw_hands) >= self.max_num_hands:
                        break
            return raw_hands

        return []


def save_clean_snapshot(clean_frame: np.ndarray, captures_dir: Path = CAPTURES_DIR) -> Path:
    """Save a clean PNG snapshot (without debug HUD bars or hand debug lines) to `captures/`."""
    captures_dir.mkdir(parents=True, exist_ok=True)
    now_dt = datetime.now()
    ms = now_dt.microsecond // 1000
    base_name = f"marvel_snap_{now_dt.strftime('%Y%m%d_%H%M%S')}_{ms:03d}"
    out_path = captures_dir / f"{base_name}.png"
    suffix = 1
    while out_path.exists():
        out_path = captures_dir / f"{base_name}_{suffix}.png"
        suffix += 1
    if not cv2.imwrite(str(out_path), clean_frame):
        raise RuntimeError(f"Failed to write clean snapshot to {out_path}")
    return out_path


class _CleanVideoRecorder:
    """Non-blocking MP4 recorder with a bounded queue and explicit drop-newest policy.

    Ensures disk encoding never blocks live hand tracking or caps memory growth.
    """

    def __init__(
        self,
        captures_dir: Path = CAPTURES_DIR,
        fps: float = float(TARGET_FPS),
        max_queue_size: int = 24,
    ) -> None:
        self.captures_dir = captures_dir
        self.fps = max(1.0, float(fps))
        self.max_queue_size = max(1, int(max_queue_size))
        self.is_recording: bool = False
        self.output_path: Optional[Path] = None
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = 0.0

        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=self.max_queue_size)
        self._worker: Optional[threading.Thread] = None

    @property
    def queue_size(self) -> int:
        """Current number of frames buffered for asynchronous encoding."""
        return self._queue.qsize()

    def _writer_loop(self, writer: cv2.VideoWriter, size: Tuple[int, int], frame_queue: queue.Queue) -> None:
        try:
            while True:
                item = frame_queue.get()
                if item is None:
                    frame_queue.task_done()
                    break
                fh, fw = item.shape[:2]
                frame_to_write = item if (fw, fh) == size else cv2.resize(item, size)
                writer.write(frame_to_write)
                self.frame_count += 1
                frame_queue.task_done()
        finally:
            writer.release()

    def toggle(self, frame_w: int, frame_h: int) -> Tuple[bool, Optional[Path]]:
        """Start background recording if idle, or stop and flush the queue if active."""
        if self.is_recording:
            return False, self.stop()

        self.captures_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.captures_dir / f"marvel_rec_{stamp}.mp4"
        suffix = 1
        while out_path.exists():
            out_path = self.captures_dir / f"marvel_rec_{stamp}_{suffix}.mp4"
            suffix += 1

        size = (int(frame_w), int(frame_h))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to initialize MP4 VideoWriter at {out_path}")

        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._writer = writer
        self._size = size
        self.output_path = out_path
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.perf_counter()
        self.is_recording = True

        self._worker = threading.Thread(
            target=self._writer_loop, args=(writer, size, self._queue), daemon=True, name="MarvelVideoRecorder"
        )
        self._worker.start()
        return True, out_path

    def write(self, clean_frame: np.ndarray) -> None:
        """Non-blocking enqueue of a clean frame snapshot; drops newest frame if queue is full."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(clean_frame.copy())
        except queue.Full:
            self.dropped_frames += 1

    def stop(self) -> Optional[Path]:
        """Flush every accepted frame and finalize the recording before returning."""
        saved_path = self.output_path
        if self._worker is not None:
            self.is_recording = False
            # Wait for queue space instead of silently deleting an accepted frame.
            if self._worker.is_alive():
                while self._worker.is_alive():
                    try:
                        self._queue.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            self._worker.join()
            self._worker = None
            self._writer = None
        self._size = None
        return saved_path


class _StableHandIdentity:
    """Associate detections with previous wrist positions, independent of detection order."""

    def __init__(self, max_distance_fraction: float = 0.22, grace_seconds: float = 0.16) -> None:
        self.max_distance_fraction = max_distance_fraction
        self.grace_seconds = grace_seconds
        self._tracks: Dict[str, Tuple[np.ndarray, float]] = {}
        self._next_id = 0

    def assign(self, hands: List[Tuple[object, str]], width: int, height: int,
               now: float) -> List[str]:
        self._tracks = {key: (pos, seen) for key, (pos, seen) in self._tracks.items()
                        if now - seen <= self.grace_seconds}
        wrists = []
        for landmarks, _ in hands:
            pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
            wrist = pts[0]
            x = float(wrist.x) if hasattr(wrist, "x") else float(wrist[0])
            y = float(wrist.y) if hasattr(wrist, "y") else float(wrist[1])
            wrists.append(np.array([x * width, y * height], dtype=np.float64))
        threshold = self.max_distance_fraction * float(np.hypot(width, height))
        candidates = sorted((float(np.linalg.norm(pos - wrist)), idx, key)
                            for idx, wrist in enumerate(wrists)
                            for key, (pos, _) in self._tracks.items())
        assigned: Dict[int, str] = {}
        used = set()
        for distance, idx, key in candidates:
            if distance <= threshold and idx not in assigned and key not in used:
                assigned[idx] = key
                used.add(key)
        result = []
        for idx, wrist in enumerate(wrists):
            key = assigned.get(idx)
            if key is None:
                key = f"track_{self._next_id}"
                self._next_id += 1
            self._tracks[key] = (wrist, now)
            result.append(key)
        return result

    def active_keys(self) -> Set[str]:
        return set(self._tracks)


class FrameProfiler:
    """Rolling stage-by-stage performance profiler for optional diagnostics HUD & benchmarks."""

    def __init__(self, window: int = 60) -> None:
        self._totals_ms: Deque[float] = deque(maxlen=window)
        self._ema: Dict[str, float] = {}

    def record(self, timings: PipelineTimings) -> None:
        """Ingest one frame's stage timings with zero allocation overhead."""
        self._totals_ms.append(timings.total_frame_ms)
        fields = (
            ("capture_ms", timings.capture_ms),
            ("color_convert_ms", timings.color_convert_ms),
            ("hand_detect_ms", timings.hand_detect_ms),
            ("face_detect_ms", timings.face_detect_ms),
            ("portal_geom_ms", timings.portal_geom_ms),
            ("portal_render_ms", timings.portal_render_ms),
            ("comic_shader_ms", timings.comic_shader_ms),
            ("face_filter_ms", timings.face_filter_ms),
            ("record_ms", timings.record_ms),
            ("total_frame_ms", timings.total_frame_ms),
        )
        for key, val in fields:
            prev = self._ema.get(key)
            self._ema[key] = val if prev is None else (0.15 * val + 0.85 * prev)

    def summary(self, dropped_frames: int = 0) -> Dict[str, float]:
        """Return rolling averages, min FPS, and queue drops for optional diagnostics display."""
        if not self._totals_ms:
            return {"avg_fps": 0.0, "min_fps": 0.0, "dropped_frames": float(dropped_frames)}
        mean_ms = max(sum(self._totals_ms) / len(self._totals_ms), 1e-4)
        max_ms = max(max(self._totals_ms), 1e-4)
        out = dict(self._ema)
        out["avg_fps"] = 1000.0 / mean_ms
        out["min_fps"] = 1000.0 / max_ms
        out["dropped_frames"] = float(dropped_frames)
        return out


def _draw_recording_indicator(frame: np.ndarray, elapsed_s: float) -> None:
    """Draw a minimal pulsing red REC dot on the live preview window while recording."""
    cx, cy = 24, 24
    pulse = (int(elapsed_s * 2.0) % 2) == 0
    cv2.circle(frame, (cx, cy), 8, (18, 18, 22), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, (40, 40, 245) if pulse else (25, 25, 175), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "REC",
        (cx + 14, cy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 250),
        1,
        cv2.LINE_AA,
    )


def run_benchmark(num_frames: int = 30) -> int:
    """Benchmark the optimized pipeline headlessly over N 1280x720 frames."""
    import os

    w, h = TARGET_WIDTH, TARGET_HEIGHT
    base = np.random.randint(30, 220, (h, w, 3), dtype=np.uint8)
    renderer = MarvelPortalRenderer()
    detector = ZeroErrorPinchDetector()
    geom_engine = renderer.geometry_engine

    def _bm_hand(wrist_x: float, thumb_x: float, idx_x: float, y_top: float, y_bot: float) -> List[Tuple[float, float]]:
        pts = [(wrist_x, 0.70) for _ in range(21)]
        pts[0] = (wrist_x, 0.70)
        pts[4] = (thumb_x, y_bot)
        pts[5] = (thumb_x, 0.52)
        pts[8] = (idx_x, y_top)
        pts[9] = (wrist_x, 0.50)
        pts[12] = ((thumb_x + idx_x) * 0.5, 0.30)
        pts[17] = (idx_x, 0.54)
        return pts

    h0 = _bm_hand(0.25, 0.20, 0.20, 0.20, 0.65)
    h1 = _bm_hand(0.75, 0.80, 0.80, 0.20, 0.65)

    tracemalloc.start()
    with (
        tempfile.TemporaryDirectory() as tmp_dir_str,
        _UnifiedHandTracker(renderer.assets_dir, max_num_hands=2, min_detection_confidence=0.30, min_presence_confidence=0.30, min_tracking_confidence=0.30) as tr,
    ):
        rec = _CleanVideoRecorder(Path(tmp_dir_str), fps=30.0, max_queue_size=24)
        rec.toggle(w, h)
        for _ in range(3):
            tr.process(base)

        totals, rgb_t, hand_t, geom_t, ren_t, shader_t, rec_t = [], [], [], [], [], [], []
        cpu_start = time.process_time()
        wall_start = time.perf_counter()

        for i in range(num_frames):
            timings = PipelineTimings()
            t0 = time.perf_counter()
            frame = cv2.flip(base, 1)
            t1 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()
            _ = tr.process(rgb)
            t3 = time.perf_counter()
            detected = [(h0, "Left"), (h1, "Right")]
            geom = geom_engine.compute_geometry(detected, w, h)
            t4 = time.perf_counter()
            detector.process_hand(h0, "Left", w, h, now=float(i) * 0.033, track_key="Left")
            detector.process_hand(h1, "Right", w, h, now=float(i) * 0.033, track_key="Right")
            out, _ = renderer.render_ar_portal_scene(
                frame, detected, portal_geometry=geom, in_place=True, timings=timings
            )
            t5 = time.perf_counter()
            rec.write(out)
            t6 = time.perf_counter()

            rgb_t.append((t2 - t1) * 1000.0)
            hand_t.append((t3 - t2) * 1000.0)
            geom_t.append((t4 - t3) * 1000.0)
            ren_t.append((t5 - t4) * 1000.0)
            shader_t.append(timings.comic_shader_ms)
            rec_t.append((t6 - t5) * 1000.0)
            totals.append((t6 - t0) * 1000.0)

        cpu_elapsed = time.process_time() - cpu_start
        wall_elapsed = max(time.perf_counter() - wall_start, 1e-6)
        rec.stop()

    current_py_mem, peak_py_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_pct = (cpu_elapsed / (wall_elapsed * max(1, os.cpu_count() or 1))) * 100.0

    print(f"OPT_RGB_MS={np.mean(rgb_t):.3f}")
    print(f"OPT_HAND_MS={np.mean(hand_t):.3f}")
    print(f"OPT_GEOM_MS={np.mean(geom_t):.3f}")
    print(f"OPT_SHADER_MS={np.mean(shader_t):.3f}")
    print(f"OPT_RENDER_MS={np.mean(ren_t):.3f}")
    print(f"OPT_ASYNC_REC_MS={np.mean(rec_t):.3f}")
    print(f"OPT_TOTAL_MS={np.mean(totals):.3f}")
    print(f"OPT_AVG_FPS={1000.0 / np.mean(totals):.2f}")
    print(f"OPT_MIN_FPS={1000.0 / np.max(totals):.2f}")
    print(f"OPT_CPU_PCT={cpu_pct:.1f}")
    print(f"OPT_PY_PEAK_MB={peak_py_mem / (1024.0 * 1024.0):.2f}")
    return 0


def _log_portal_geometry_diagnostic(
    geom: object,
    timings: PipelineTimings,
) -> None:
    """Log the 4 raw fingertip coordinates, hand-center distance, and rendered portal bbox."""
    if not getattr(geom, "is_valid", False):
        return
    p1 = geom.p1_left_thumb
    p2 = geom.p2_left_index
    p3 = geom.p3_right_index
    p4 = geom.p4_right_thumb
    left_cx = 0.5 * (p1[0] + p2[0])
    right_cx = 0.5 * (p4[0] + p3[0])
    raw_portal_w = abs(right_cx - left_cx)
    bx1, by1, bx2, by2 = geom.bbox
    rx1, ry1, rx2, ry2 = timings.rendered_bbox
    print(
        f"[DIAG] L4={p1} L8={p2} R8={p3} R4={p4} | "
        f"L_cx={left_cx:.1f} R_cx={right_cx:.1f} raw_dist={raw_portal_w:.1f}px | "
        f"geom_bbox=({bx2 - bx1}x{by2 - by1}) rendered_bbox=({rx2 - rx1}x{ry2 - ry1}) "
        f"dest_wh={timings.dest_wh} corners={geom.quad.tolist()}"
    )


def run_app(
    camera_index: Optional[int] = None,
    enable_profile: bool = False,
    enable_face_filter: bool = False,
    enable_diag: bool = False,
) -> int:
    """Launch the live webcam Marvel AR Portal & Zero-Error Gesture Controller."""
    cap, backend_badge = discover_and_select_camera(cli_camera_index=camera_index)
    if not cap.isOpened():
        req = f"--camera {camera_index}" if camera_index is not None else "interactive scan"
        print(
            f"Error: Unable to open any working camera ({req}; last fallback: {backend_badge}).",
            file=sys.stderr,
        )
        return 1

    renderer = MarvelPortalRenderer()
    geom_engine = renderer.geometry_engine
    detector = ZeroErrorPinchDetector(
        trigger_threshold=0.29,
        release_threshold=0.38,
        ema_alpha=0.82,
        required_consecutive_frames=1,
        cooldown_seconds=0.25,
        index_extension_factor=0.85,
        index_cancel_factor=0.78,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)

    prev_time = time.perf_counter()
    fps_ema = 0.0
    show_diagnostics = bool(enable_profile)
    face_filter_active = bool(enable_face_filter)
    diag_log_active = bool(enable_diag)
    snapshot_requested = False

    profiler = FrameProfiler(window=60)
    recorder = _CleanVideoRecorder(captures_dir=CAPTURES_DIR, fps=float(TARGET_FPS), max_queue_size=24)
    hand_identity = _StableHandIdentity()
    face_tracker = _OptionalFaceTracker(renderer.assets_dir, inference_interval=3)

    try:
        with _UnifiedHandTracker(
            assets_dir=renderer.assets_dir,
            max_num_hands=2,
            min_detection_confidence=0.30,
            min_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        ) as hand_tracker:
            while True:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                timings = PipelineTimings()
                t_frame_start = time.perf_counter()

                ok, raw_frame = cap.read()
                if not ok or raw_frame is None:
                    break

                # Mirror horizontally so Left/Right match user perspective naturally
                frame = cv2.flip(raw_frame, 1)
                frame_h, frame_w = frame.shape[:2]
                t_after_cap = time.perf_counter()
                timings.capture_ms = (t_after_cap - t_frame_start) * 1000.0

                now = t_after_cap
                dt = max(now - prev_time, 1e-6)
                prev_time = now
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.15 * inst_fps + 0.85 * fps_ema)

                # Single BGR->RGB conversion shared cleanly across detection tasks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t_after_rgb = time.perf_counter()
                timings.color_convert_ms = (t_after_rgb - t_after_cap) * 1000.0

                # 1. Critical Path: Stateless Hand Detection
                detected_hands = hand_tracker.process(rgb_frame)
                t_after_hand = time.perf_counter()
                timings.hand_detect_ms = (t_after_hand - t_after_rgb) * 1000.0

                # 2. Independent Portal Geometry Calculation (ONLY Left/Right Thumb 4 & Index 8)
                portal_geom = geom_engine.compute_geometry(detected_hands, frame_w, frame_h)
                t_after_geom = time.perf_counter()
                timings.portal_geom_ms = (t_after_geom - t_after_hand) * 1000.0

                # 3. Independent Character-Selection Gesture Engine (Thumb 4 + Middle 12)
                telemetries: List[HandTelemetry] = []
                active_keys: Set[str] = set()
                stable_keys = hand_identity.assign(detected_hands, frame_w, frame_h, now)
                for idx, (hand_lms, hand_label) in enumerate(detected_hands):
                    track_key = stable_keys[idx]
                    active_keys.add(track_key)
                    telemetry = detector.process_hand(
                        hand_landmarks=hand_lms,
                        hand_label=hand_label,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        now=now,
                        track_key=track_key,
                    )
                    telemetries.append(telemetry)
                    if telemetry.triggered_this_frame:
                        switched = renderer.switch_character(step=1, source_label=hand_label)
                        renderer.trigger_shockwave(telemetry.midpoint_px, switched.accent_bgr)
                detector.reset_missing_hands(hand_identity.active_keys())

                # 4. Optional Throttled ROI Face Detection (never blocks/halts hand tracking)
                face_detections: Optional[List[List[Tuple[float, float]]]] = None
                if face_filter_active and portal_geom.is_valid:
                    t_f0 = time.perf_counter()
                    face_detections = face_tracker.process_roi(rgb_frame, portal_geom.bbox, frame_w, frame_h)
                    timings.face_detect_ms = (time.perf_counter() - t_f0) * 1000.0

                # 5. Zero-Copy In-Place ROI Portal Rendering
                t_r0 = time.perf_counter()
                frame, portal_mode = renderer.render_ar_portal_scene(
                    base_frame=frame,
                    hand_landmarks_list=detected_hands,
                    face_landmarks_list=face_detections,
                    portal_geometry=portal_geom,
                    in_place=True,
                    timestamp=now,
                    timings=timings,
                )
                timings.portal_render_ms = (time.perf_counter() - t_r0) * 1000.0
                if diag_log_active and portal_geom.is_valid:
                    _log_portal_geometry_diagnostic(portal_geom, timings)

                # Lazy snapshot/recording capture (zero full-frame copy during normal live frames)
                if snapshot_requested:
                    snapshot_requested = False
                    try:
                        snap_path = save_clean_snapshot(frame, captures_dir=CAPTURES_DIR)
                        print(f"[CAPTURE] Saved clean snapshot: {snap_path}")
                    except Exception as exc:
                        print(f"[CAPTURE] Snapshot error: {exc}", file=sys.stderr)

                if recorder.is_recording:
                    t_rec0 = time.perf_counter()
                    recorder.write(frame)
                    timings.record_ms = (time.perf_counter() - t_rec0) * 1000.0

                if len(detected_hands) < 2:
                    renderer.render_fingertip_dots(frame, detected_hands)

                timings.total_frame_ms = (time.perf_counter() - t_frame_start) * 1000.0
                profiler.record(timings)

                if recorder.is_recording:
                    _draw_recording_indicator(frame, now - recorder.start_time)
                else:
                    renderer.render_shockwaves(frame)
                    diag_payload = profiler.summary(recorder.dropped_frames) if show_diagnostics else None
                    renderer.render_hud(
                        frame=frame,
                        os_backend_badge=backend_badge,
                        fps=fps_ema,
                        portal_mode=portal_mode,
                        telemetries=telemetries,
                        diagnostics=diag_payload,
                    )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    renderer.reset_character()
                elif key in (ord("p"), ord("P")):
                    show_diagnostics = not show_diagnostics
                elif key in (ord("d"), ord("D")):
                    diag_log_active = not diag_log_active
                    print(f"[DIAG] Portal geometry logger {'ENABLED' if diag_log_active else 'DISABLED'}")
                elif key in (ord("f"), ord("F")):
                    face_filter_active = not face_filter_active
                    print(f"[FACE] Comic face overlay {'ENABLED' if face_filter_active else 'DISABLED'}")
                elif key in (ord("s"), ord("S")):
                    snapshot_requested = True
                elif key in (ord("v"), ord("V")):
                    try:
                        started, video_path = recorder.toggle(frame_w, frame_h)
                        if started and video_path is not None:
                            print(f"[REC] Recording async clean MP4 -> {video_path}")
                        elif video_path is not None:
                            print(
                                f"[REC] Saved clean video ({recorder.frame_count} frames, "
                                f"{recorder.dropped_frames} dropped): {video_path}"
                            )
                    except Exception as exc:
                        print(f"[REC] Video capture error: {exc}", file=sys.stderr)
    finally:
        if recorder.is_recording:
            final_video = recorder.stop()
            if final_video is not None:
                print(
                    f"[REC] Saved clean video ({recorder.frame_count} frames, "
                    f"{recorder.dropped_frames} dropped): {final_video}"
                )
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line options for live execution, profiling, or self-testing."""
    parser = argparse.ArgumentParser(description="Marvel Portal & Gesture Controller")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Webcam device index (skips interactive terminal camera selection if specified)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable optional real-time performance diagnostics HUD overlay",
    )
    parser.add_argument(
        "--face-filter",
        action="store_true",
        help="Enable optional throttled ROI Marvel comic face filter inside the portal",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Log raw 4-point hand coordinates, portal width, and rendered bbox per frame",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a 30-frame headless 1280x720 performance benchmark and exit",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run headless verification of all 12 mandatory acceptance tests.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        sys.exit(run_self_test())
    if args.benchmark:
        sys.exit(run_benchmark())
    sys.exit(
        run_app(
            camera_index=args.camera,
            enable_profile=args.profile,
            enable_face_filter=args.face_filter,
            enable_diag=args.diag,
        )
    )