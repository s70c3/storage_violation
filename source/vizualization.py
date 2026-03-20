from __future__ import annotations

import time
import logging
from pathlib import Path

import cv2
import numpy as np


def rgb01_to_bgr_u8(rgb01: np.ndarray) -> np.ndarray:
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01[..., ::-1] * 255.0).astype(np.uint8)


def mask01_to_bgr_u8(mask01: np.ndarray) -> np.ndarray:
    if mask01 is None:
        return np.zeros((1, 1, 3), np.uint8)
    m = np.clip(mask01, 0.0, 1.0)
    m_u8 = (m * 255.0).astype(np.uint8)
    return cv2.cvtColor(m_u8, cv2.COLOR_GRAY2BGR)


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    for b in boxes_xyxy:
        x1, y1, x2, y2 = map(int, np.asarray(b).tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_overlay(base_bgr: np.ndarray, top_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if base_bgr.shape[:2] != top_bgr.shape[:2]:
        top_bgr = cv2.resize(
            top_bgr,
            (base_bgr.shape[1], base_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.addWeighted(base_bgr, 1.0 - alpha, top_bgr, alpha, 0.0)


def resize_thumb(img: np.ndarray, max_w: int = 420, max_h: int = 240) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_size(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )


def save_rt_panel(
    camera_id: str,
    iter_idx: int,
    empty_rgb01: np.ndarray,
    recent_rgb01: np.ndarray,
    cur_rgb01: np.ndarray,
    cd_mask01: np.ndarray,
    pending_boxes: np.ndarray | list,
    reported_boxes: np.ndarray | list,
    logger: logging.Logger | None = None,
    out_dir: str | Path = "/tmp/rt_tests",
) -> None:
    try:
        logger = logger or logging.getLogger("StorageViolationVisualization")
        rt_dir = Path(out_dir)
        rt_dir.mkdir(parents=True, exist_ok=True)

        ideal_bgr = rgb01_to_bgr_u8(empty_rgb01)
        recent_bgr = rgb01_to_bgr_u8(recent_rgb01)
        cur_bgr = rgb01_to_bgr_u8(cur_rgb01)
        cd_bgr = mask01_to_bgr_u8(cd_mask01)

        cur_bgr_boxed = draw_boxes(cur_bgr, pending_boxes, color=(0, 0, 255), thickness=2)
        cur_bgr_boxed = draw_boxes(cur_bgr_boxed, reported_boxes, color=(0, 255, 0), thickness=2)

        cd_bgr_boxed = draw_boxes(cd_bgr, pending_boxes, color=(0, 0, 255), thickness=2)
        cd_bgr_boxed = draw_boxes(cd_bgr_boxed, reported_boxes, color=(0, 255, 0), thickness=2)

        overlay_bgr = make_overlay(ideal_bgr, cur_bgr, alpha=0.5)
        overlay_bgr_boxed = draw_boxes(overlay_bgr, pending_boxes, color=(0, 0, 255), thickness=2)
        overlay_bgr_boxed = draw_boxes(overlay_bgr_boxed, reported_boxes, color=(0, 255, 0), thickness=2)

        tiles = [
            put_label(ideal_bgr, "ideal"),
            put_label(recent_bgr, "recent"),
            put_label(cur_bgr_boxed, f"current (pending={len(pending_boxes)}, reported={len(reported_boxes)})"),
            put_label(cd_bgr_boxed, "cd"),
            put_label(overlay_bgr_boxed, "ideal + current"),
            np.full_like(ideal_bgr, 20),
        ]

        thumbs = [resize_thumb(t, max_w=420, max_h=240) for t in tiles[:6]]
        target_h = max(t.shape[0] for t in thumbs)
        target_w = max(t.shape[1] for t in thumbs)
        thumbs = [pad_to_size(t, target_w, target_h) for t in thumbs]

        spacer = np.full((target_h, 12, 3), 20, dtype=np.uint8)
        hspacer = np.full((12, target_w * 3 + 24, 3), 20, dtype=np.uint8)

        row1 = cv2.hconcat([thumbs[0], spacer, thumbs[1], spacer, thumbs[2]])
        row2 = cv2.hconcat([thumbs[3], spacer, thumbs[4], spacer, thumbs[5]])
        panel = cv2.vconcat([row1, hspacer, row2])

        ts_ms = int(time.time() * 1000)
        out_path = rt_dir / f"cam_{camera_id}_it_{iter_idx:06d}_{ts_ms}.jpg"

        ok = cv2.imwrite(str(out_path), panel)
        if not ok:
            logger.warning(f"[RT_SAVE] cv2.imwrite returned False: {out_path}")

    except Exception as e:
        if logger is not None:
            logger.warning(f"[RT_SAVE] failed to save debug panel: {e}")