from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import supervision as sv


@dataclass
class TrackState:
    first_seen: float
    last_seen: float
    anchor_bbox: np.ndarray
    stationary_started: float
    is_stationary: bool = False


class ByteTrackStationaryTracker:
    """
    Ready-made ByteTrack + simple stationary logic on top of tracker_id.

    candidate_tracks:
        visible now, but not stationary long enough yet

    reported_tracks:
        visible now and stationary long enough
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

    def _moved_too_much(self, anchor_bbox: np.ndarray, cur_bbox: np.ndarray) -> bool:
        c0, d0 = self._center_and_diag(anchor_bbox)
        c1, _ = self._center_and_diag(cur_bbox)

        shift_px = float(np.linalg.norm(c1 - c0))
        shift_norm = shift_px / d0

        return (shift_px > self.max_center_shift_px) or (shift_norm > self.max_center_shift_norm)

    def update(
        self,
        now: float,
        current_bboxes: List[np.ndarray],
        current_masks: List[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            candidate_boxes: visible now, not stationary yet
            reported_boxes: visible now, stationary long enough
        """
        if current_masks is None:
            current_masks = [None] * len(current_bboxes)

        if len(current_masks) != len(current_bboxes):
            raise ValueError("current_masks/current_bboxes must have same length")

        if len(current_bboxes) == 0:
            # прогоняем пустые детекции через трекер, чтобы он обновил свое внутреннее состояние
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )
            tracked = self.tracker.update_with_detections(empty)
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0, 4), dtype=np.int32),
            )

        xyxy = np.asarray(current_bboxes, dtype=np.float32).reshape(-1, 4)
        conf = np.ones((len(current_bboxes),), dtype=np.float32)
        class_id = np.zeros((len(current_bboxes),), dtype=int)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id,
            mask=np.asarray(current_masks, dtype=object) if current_masks else None,
        )

        tracked = self.tracker.update_with_detections(detections)

        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            return (
                np.empty((0, 4), dtype=np.int32),
                np.empty((0, 4), dtype=np.int32),
            )

        visible_ids: set[int] = set()
        candidate_boxes: List[np.ndarray] = []
        reported_boxes: List[np.ndarray] = []

        for i in range(len(tracked.xyxy)):
            tid = tracker_ids[i]
            if tid is None:
                continue

            tid = int(tid)
            bbox = np.asarray(tracked.xyxy[i], dtype=np.float32)
            visible_ids.add(tid)

            st = self._states.get(tid)
            if st is None:
                st = TrackState(
                    first_seen=now,
                    last_seen=now,
                    anchor_bbox=bbox.copy(),
                    stationary_started=now,
                    is_stationary=False,
                )
                self._states[tid] = st
            else:
                st.last_seen = now
                if self._moved_too_much(st.anchor_bbox, bbox):
                    st.anchor_bbox = bbox.copy()
                    st.stationary_started = now
                    st.is_stationary = False
                else:
                    st.is_stationary = (now - st.stationary_started) >= self.stationary_time_sec

            if st.is_stationary:
                reported_boxes.append(bbox.astype(np.int32))
            else:
                candidate_boxes.append(bbox.astype(np.int32))

        # Чистим состояния треков, которых больше нет у ByteTrack
        stale_ids = [tid for tid in self._states.keys() if tid not in visible_ids]
        for tid in stale_ids:
            self._states.pop(tid, None)

        cand = np.stack(candidate_boxes, axis=0) if candidate_boxes else np.empty((0, 4), dtype=np.int32)
        rep = np.stack(reported_boxes, axis=0) if reported_boxes else np.empty((0, 4), dtype=np.int32)
        return cand, rep