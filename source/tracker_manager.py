from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    anchor_bbox: np.ndarray
    prev_bbox: np.ndarray
    stationary_started: float
    is_stationary: bool = False


class ByteTrackStationaryTracker:
    """
    Returns:
        candidate_boxes:
            all current detections that are NOT stationary yet

        reported_boxes:
            current detections that are already stationary
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25,
        minimum_consecutive_frames: int = 1,
        stationary_time_sec: float = 5.0,
        max_center_shift_px: float = 20.0,
        max_center_shift_norm: float = 0.08,
        logger: logging.Logger | None = None,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        self.stationary_time_sec = float(stationary_time_sec)
        self.max_center_shift_px = float(max_center_shift_px)
        self.max_center_shift_norm = float(max_center_shift_norm)

        self.lost_track_buffer = int(lost_track_buffer)
        self.frame_rate = max(1, int(frame_rate))
        self.state_ttl_sec = max(1.0, float(self.lost_track_buffer) / float(self.frame_rate))

        self._states: Dict[int, TrackState] = {}
        self._logger = logger or logging.getLogger("ByteTrackStationaryTracker")

    def reset(self) -> None:
        self.tracker.reset()
        self._states.clear()

    @staticmethod
    def _center_and_diag(b: np.ndarray) -> tuple[np.ndarray, float]:
        b = np.asarray(b, dtype=np.float32).reshape(4)
        c = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
        diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
        return c, max(diag, 1e-6)

    def _moved_too_much(self, ref_bbox: np.ndarray, cur_bbox: np.ndarray) -> bool:
        c0, d0 = self._center_and_diag(ref_bbox)
        c1, _ = self._center_and_diag(cur_bbox)

        shift_px = float(np.linalg.norm(c1 - c0))
        shift_norm = shift_px / d0

        return (shift_px > self.max_center_shift_px) or (shift_norm > self.max_center_shift_norm)

    @staticmethod
    def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        if union <= 1e-6:
            return 0.0
        return float(inter / union)

    def _current_minus_reported(
        self,
        current_boxes: np.ndarray,
        reported_boxes: np.ndarray,
        iou_thr: float = 0.3,
    ) -> np.ndarray:
        if len(current_boxes) == 0:
            return np.empty((0, 4), dtype=np.int32)

        if len(reported_boxes) == 0:
            return current_boxes.astype(np.int32, copy=False)

        candidate = []
        for cur_box in current_boxes:
            matched_reported = False
            for rep_box in reported_boxes:
                if self._bbox_iou_xyxy(cur_box, rep_box) >= float(iou_thr):
                    matched_reported = True
                    break
            if not matched_reported:
                candidate.append(np.asarray(cur_box, dtype=np.int32))

        if not candidate:
            return np.empty((0, 4), dtype=np.int32)

        return np.stack(candidate, axis=0)

    def _cleanup_stale_states(self, now: float) -> None:
        stale_ids = [
            tid
            for tid, st in self._states.items()
            if (now - st.last_seen) > self.state_ttl_sec
        ]
        for tid in stale_ids:
            self._states.pop(tid, None)

    def update(
        self,
        now: float,
        current_bboxes: List[np.ndarray],
        current_masks: List[np.ndarray] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        if current_masks is None:
            current_masks = [None] * len(current_bboxes)

        if len(current_masks) != len(current_bboxes):
            raise ValueError("current_masks/current_bboxes must have same length")

        if len(current_bboxes) == 0:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )
            _ = self.tracker.update_with_detections(empty)
            self._cleanup_stale_states(now)
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0, 4), dtype=np.int32),
                [],
                [],
            )

        current_boxes = np.asarray(current_bboxes, dtype=np.int32).reshape(-1, 4)

        detections = sv.Detections(
            xyxy=current_boxes.astype(np.float32, copy=False),
            confidence=np.ones((len(current_boxes),), dtype=np.float32),
            class_id=np.zeros((len(current_boxes),), dtype=int),
            mask=np.asarray(current_masks, dtype=object) if current_masks else None,
        )

        tracked = self.tracker.update_with_detections(detections)

        reported_boxes: List[np.ndarray] = []
        reported_ids: List[int] = []

        tracker_ids = tracked.tracker_id
        if tracker_ids is not None:
            for i in range(len(tracked.xyxy)):
                tid = tracker_ids[i]
                if tid is None:
                    continue

                tid = int(tid)
                bbox = np.asarray(tracked.xyxy[i], dtype=np.float32)

                st = self._states.get(tid)
                if st is None:
                    st = TrackState(
                        first_seen=now,
                        last_seen=now,
                        anchor_bbox=bbox.copy(),
                        prev_bbox=bbox.copy(),
                        stationary_started=now,
                        is_stationary=False,
                    )
                    self._states[tid] = st
                else:
                    moved_from_anchor = self._moved_too_much(st.anchor_bbox, bbox)
                    moved_from_prev = self._moved_too_much(st.prev_bbox, bbox)

                    st.last_seen = now

                    if moved_from_anchor or moved_from_prev:
                        st.anchor_bbox = bbox.copy()
                        st.stationary_started = now
                        st.is_stationary = False
                    else:
                        st.is_stationary = (now - st.stationary_started) >= self.stationary_time_sec

                    st.prev_bbox = bbox.copy()

                if st.is_stationary:
                    reported_boxes.append(np.asarray(bbox, dtype=np.int32))
                    reported_ids.append(tid)

        self._cleanup_stale_states(now)

        reported_arr = (
            np.stack(reported_boxes, axis=0)
            if reported_boxes
            else np.empty((0, 4), dtype=np.int32)
        )

        candidate_arr = self._current_minus_reported(
            current_boxes=current_boxes,
            reported_boxes=reported_arr,
            iou_thr=0.3,
        )

        candidate_ids = self._candidate_track_ids(
            candidate_arr=candidate_arr,
            current_boxes=current_boxes,
            tracker_ids=tracker_ids,
        )

        return candidate_arr, reported_arr, candidate_ids, reported_ids

    def _candidate_track_ids(
        self,
        candidate_arr: np.ndarray,
        current_boxes: np.ndarray,
        tracker_ids: np.ndarray | None,
    ) -> List[int]:
        """Сопоставляет строки candidate_arr с детекциями ByteTrack и возвращает tracker_id."""
        if len(candidate_arr) == 0:
            return []
        out: List[int] = []
        for row in candidate_arr:
            best_j = -1
            best_iou = 0.0
            r = np.asarray(row, dtype=np.float32).reshape(4)
            for j in range(len(current_boxes)):
                iou = self._bbox_iou_xyxy(
                    np.asarray(current_boxes[j], dtype=np.float32),
                    r,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if (
                best_j >= 0
                and best_iou >= 0.25
                and tracker_ids is not None
                and best_j < len(tracker_ids)
            ):
                raw = tracker_ids[best_j]
                out.append(int(raw) if raw is not None else -1)
            else:
                out.append(-1)
        return out