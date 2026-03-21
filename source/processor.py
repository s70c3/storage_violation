from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .segmentator import SemanticSegmentatorProtocol
from .tracker_manager import ByteTrackStationaryTracker
from .visualization import save_rt_panel


@dataclass
class CameraEMAState:
    empty_rgb01: np.ndarray
    recent_rgb01: np.ndarray
    last_ts_mono: float
    last_detect_mono: float
    detect_started: bool = False
    recent_frame_i: int = 0
    recent_dt_accum: float = 0.0


class StorageViolationFrameProcessor:
    """
    candidate_boxes / pending_candidate_boxes:
        visible now, but not stationary long enough yet (GREEN)

    reported_boxes / boxes:
        visible now and stationary long enough (RED)

    All internal processing is done in resized resolution.
    If long side > max_long_side, frame and ideal are resized to that limit.
    """

    def __init__(
        self,
        cd_segmentator: SemanticSegmentatorProtocol,
        delta: int = 2,
        ema_tau_sec: float = 5.0,
        ema_min_alpha: float = 0.02,
        ema_max_alpha: float = 0.12,
        min_empty_weight: float = 0.20,
        thr: float = 0.5,
        min_side: int = 100,
        morph_ksize: int = 3,
        stationary_time_sec: float = 5.0,
        tracker_track_activation_threshold: float = 0.25,
        tracker_lost_track_buffer: int = 10,
        tracker_minimum_matching_threshold: float = 0.7,
        tracker_frame_rate: int = 25,
        tracker_minimum_consecutive_frames: int = 1,
        tracker_max_center_shift_px: float = 20.0,
        tracker_max_center_shift_norm: float = 0.08,
        recent_update_every: int = 10,
        max_long_side: int = 640,
        logger: logging.Logger | None = None,
    ):
        self.cd_segmentator = cd_segmentator
        self.delta = int(delta)

        self.camera_state: Dict[str, CameraEMAState] = {}
        self.tracker_by_camera: Dict[str, ByteTrackStationaryTracker] = {}

        self.ema_tau_sec = float(ema_tau_sec)
        self.ema_min_alpha = float(ema_min_alpha)
        self.ema_max_alpha = float(ema_max_alpha)

        self.min_empty_weight = float(min_empty_weight)

        self.thr = float(thr)
        self.min_side = int(min_side)
        self.morph_ksize = int(morph_ksize)

        self.stationary_time_sec = float(stationary_time_sec)
        self.tracker_track_activation_threshold = float(tracker_track_activation_threshold)
        self.tracker_lost_track_buffer = int(tracker_lost_track_buffer)
        self.tracker_minimum_matching_threshold = float(tracker_minimum_matching_threshold)
        self.tracker_frame_rate = int(tracker_frame_rate)
        self.tracker_minimum_consecutive_frames = int(tracker_minimum_consecutive_frames)
        self.tracker_max_center_shift_px = float(tracker_max_center_shift_px)
        self.tracker_max_center_shift_norm = float(tracker_max_center_shift_norm)

        self.recent_update_every = max(1, int(recent_update_every))
        self.max_long_side = int(max_long_side)

        self._rt_iter_by_camera: Dict[str, int] = {}
        self._prepared_empty_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

        self._logger = logger or logging.getLogger("StorageViolationProcessor")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self._cd_mean = np.asarray([0.485, 0.456, 0.406], np.float32)
        self._cd_std = np.asarray([0.229, 0.224, 0.225], np.float32)

    def update_runtime_params(
        self,
        min_side: int | None = None,
        stationary_time_sec: float | None = None,
        max_long_side: int | None = None,
    ) -> None:
        if min_side is not None:
            self.min_side = int(min_side)

        if stationary_time_sec is not None:
            self.stationary_time_sec = float(stationary_time_sec)

        if max_long_side is not None:
            self.max_long_side = int(max_long_side)
            self.camera_state.clear()
            self.tracker_by_camera.clear()
            self._rt_iter_by_camera.clear()
            self._prepared_empty_cache.clear()
            self._logger.info("[PARAM_UPDATE] camera states reset because max_long_side changed")

        self._logger.info(
            f"[PARAM_UPDATE] min_side={self.min_side} "
            f"stationary_time_sec={self.stationary_time_sec} "
            f"max_long_side={self.max_long_side}"
        )

        for camera_id in list(self.tracker_by_camera.keys()):
            self.tracker_by_camera[camera_id] = self._make_tracker()
            self._logger.info(f"[PARAM_UPDATE] tracker reset for camera={camera_id}")

    def load(self) -> None:
        self.cd_segmentator.load()

    def close(self) -> None:
        if self.cd_segmentator is not None:
            self.cd_segmentator.close()
        self.camera_state.clear()
        self.tracker_by_camera.clear()
        self._rt_iter_by_camera.clear()
        self._prepared_empty_cache.clear()

    def reset_camera(self, camera_id: str) -> None:
        camera_id = str(camera_id)
        self.camera_state.pop(camera_id, None)
        self.tracker_by_camera.pop(camera_id, None)
        self._rt_iter_by_camera.pop(camera_id, None)

        keys_to_drop = [k for k in self._prepared_empty_cache.keys() if k[0] == camera_id]
        for k in keys_to_drop:
            self._prepared_empty_cache.pop(k, None)

    def _make_tracker(self) -> ByteTrackStationaryTracker:
        return ByteTrackStationaryTracker(
            track_activation_threshold=self.tracker_track_activation_threshold,
            lost_track_buffer=self.tracker_lost_track_buffer,
            minimum_matching_threshold=self.tracker_minimum_matching_threshold,
            frame_rate=self.tracker_frame_rate,
            minimum_consecutive_frames=self.tracker_minimum_consecutive_frames,
            stationary_time_sec=self.stationary_time_sec,
            max_center_shift_px=self.tracker_max_center_shift_px,
            max_center_shift_norm=self.tracker_max_center_shift_norm,
            logger=self._logger,
        )

    def _ensure_camera_tracker(self, camera_id: str) -> None:
        if camera_id not in self.tracker_by_camera:
            self.tracker_by_camera[camera_id] = self._make_tracker()
            self._logger.info(f"[INIT] tracker created for camera={camera_id}")

    def _should_detect(self, st: CameraEMAState, now_mono: float) -> bool:
        if (now_mono - st.last_detect_mono) >= float(self.delta):
            st.last_detect_mono = now_mono
            return True
        return False

    def _ema_alpha(self, dt: float) -> float:
        if self.ema_tau_sec <= 1e-6:
            a = 1.0
        else:
            a = 1.0 - math.exp(-max(0.0, dt) / self.ema_tau_sec)
        return float(np.clip(a, self.ema_min_alpha, self.ema_max_alpha))

    @staticmethod
    def _bgr_u8_to_rgb01(bgr_u8: np.ndarray) -> np.ndarray:
        return bgr_u8[..., ::-1].astype(np.float32) / 255.0

    @staticmethod
    def _resize_bgr_if_needed(
        bgr: np.ndarray,
        max_long_side: int,
    ) -> tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        long_side = max(h, w)

        if max_long_side <= 0 or long_side <= max_long_side:
            return np.ascontiguousarray(bgr), 1.0

        scale = float(max_long_side) / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized), scale

    @staticmethod
    def _resize_polygon(
        polygon: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        if scale == 1.0:
            return np.asarray(polygon, dtype=np.float32)
        return np.asarray(polygon, dtype=np.float32) * float(scale)

    @staticmethod
    def _scale_box_xyxy(
        box: np.ndarray | list | tuple,
        scale: float,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(float, np.asarray(box).tolist())
        if scale != 1.0:
            x1 *= scale
            y1 *= scale
            x2 *= scale
            y2 *= scale

        h, w = target_hw
        x1 = int(np.clip(round(x1), 0, w))
        y1 = int(np.clip(round(y1), 0, h))
        x2 = int(np.clip(round(x2), 0, w))
        y2 = int(np.clip(round(y2), 0, h))
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def _build_zone_mask_u8(
        self,
        image_hw: Tuple[int, int],
        polygons: Optional[List[np.ndarray]],
        frame_obj: Any = None,
    ) -> np.ndarray:
        h, w = image_hw
        zone_mask = np.zeros((h, w), dtype=np.uint8)

        if polygons:
            for polygon in polygons:
                pts = np.asarray(polygon, dtype=np.int32)
                if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                    self._logger.warning(f"[ZONE] skip invalid polygon with shape={pts.shape}")
                    continue
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(zone_mask, [pts], 255)
        else:
            zone_mask[:, :] = 255

        if frame_obj is not None:
            raw_humans = getattr(frame_obj, "raw_humans", None)
            raw_human_boxes = getattr(raw_humans, "boxes", None) if raw_humans is not None else None
            if raw_human_boxes is not None:
                for box in raw_human_boxes:
                    x1, y1, x2, y2 = map(int, np.asarray(box).tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        zone_mask[y1:y2, x1:x2] = 0

        return zone_mask

    def _mask01_to_instances(self, mask01: np.ndarray) -> List[np.ndarray]:
        if mask01 is None:
            return []

        m = (mask01 >= self.thr).astype(np.uint8) * 255
        m = np.ascontiguousarray(m)

        if self.morph_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_ksize, self.morph_ksize))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = m.shape[:2]
        out: List[np.ndarray] = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            side = min(bw, bh)

            if side <= 0:
                continue
            if self.min_side > 0 and side < float(self.min_side):
                continue

            inst = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inst, [c], -1, 1, thickness=-1)
            out.append(inst.astype(bool))

        return out

    @staticmethod
    def _get_bbox(mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.array([0, 0, 0, 0], dtype=np.float32)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def _build_cd_tensor_1x9(
        self,
        empty_rgb01: np.ndarray,
        recent_rgb01: np.ndarray,
        cur_rgb01: np.ndarray,
    ) -> torch.Tensor:
        def to_chw(x: np.ndarray) -> np.ndarray:
            x = np.ascontiguousarray(x, dtype=np.float32).transpose(2, 0, 1)
            x = (x - self._cd_mean[:, None, None]) / self._cd_std[:, None, None]
            return x

        xe = to_chw(empty_rgb01)
        xr = to_chw(recent_rgb01)
        xc = to_chw(cur_rgb01)
        x9 = np.concatenate([xe, xr, xc], axis=0)
        return torch.from_numpy(x9).unsqueeze(0)

    def _update_recent(
        self,
        st: CameraEMAState,
        cur_rgb01: np.ndarray,
        now_mono: float,
    ) -> float:
        dt = now_mono - st.last_ts_mono
        st.last_ts_mono = now_mono

        st.recent_frame_i += 1
        st.recent_dt_accum += max(0.0, float(dt))

        if (st.recent_frame_i % self.recent_update_every) != 0:
            return 0.0

        dt_eff = st.recent_dt_accum
        st.recent_dt_accum = 0.0

        a = self._ema_alpha(dt_eff)
        st.recent_rgb01 = (1.0 - a) * st.recent_rgb01 + a * cur_rgb01
        return float(a)

    def _infer_cd_mask01(
        self,
        empty_rgb01: np.ndarray,
        recent_rgb01: np.ndarray,
        cur_rgb01: np.ndarray,
        camera_id: str,
    ) -> np.ndarray:
        if cur_rgb01.ndim != 3 or cur_rgb01.shape[2] != 3:
            raise ValueError(f"cur_rgb01 must be HxWx3 float32, got {cur_rgb01.shape}")

        exp_hw = (cur_rgb01.shape[0], cur_rgb01.shape[1])
        x = self._build_cd_tensor_1x9(empty_rgb01, recent_rgb01, cur_rgb01)

        pred_fn = getattr(self.cd_segmentator, "predict_semantic01", None)
        if callable(pred_fn):
            sem01 = pred_fn(x, {"camera_id": camera_id})
            sem01 = np.asarray(sem01, dtype=np.float32)

            if sem01.shape != exp_hw:
                self._logger.warning(
                    f"[CD] predict_semantic01 bad shape={sem01.shape}, expected={exp_hw}"
                )
                return np.zeros(exp_hw, np.float32)

            return sem01

        cd_res = self.cd_segmentator([x], {"camera_id": camera_id})
        if not cd_res:
            return np.zeros(exp_hw, np.float32)

        bm = getattr(cd_res[0], "binary_masks", None)
        masks = [] if (bm is None or bm.masks is None) else (bm.masks or [])
        if not masks:
            return np.zeros(exp_hw, np.float32)

        sem = np.zeros(exp_hw, dtype=bool)
        for m in masks:
            sem |= np.asarray(m, dtype=bool)

        return sem.astype(np.float32, copy=False)

    def _prepare_empty_rgb01(
        self,
        camera_id: str,
        ideal_bgr: np.ndarray,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        cache_key = (camera_id, target_hw[0], target_hw[1])
        cached = self._prepared_empty_cache.get(cache_key)
        if cached is not None:
            return cached

        ih, iw = ideal_bgr.shape[:2]
        th, tw = target_hw

        if (ih, iw) != (th, tw):
            ideal_bgr = cv2.resize(ideal_bgr, (tw, th), interpolation=cv2.INTER_AREA)

        empty_rgb01 = self._bgr_u8_to_rgb01(np.ascontiguousarray(ideal_bgr))
        self._prepared_empty_cache[cache_key] = empty_rgb01
        return empty_rgb01

    def _init_camera_state_if_needed(
        self,
        camera_id: str,
        frame_hw: Tuple[int, int],
        ideal_bgr: np.ndarray,
        now_mono: float,
    ) -> None:
        if camera_id in self.camera_state:
            return

        empty_rgb01 = self._prepare_empty_rgb01(
            camera_id=camera_id,
            ideal_bgr=ideal_bgr,
            target_hw=frame_hw,
        )

        self.camera_state[camera_id] = CameraEMAState(
            empty_rgb01=empty_rgb01,
            recent_rgb01=empty_rgb01.copy(),
            last_ts_mono=now_mono,
            last_detect_mono=now_mono - float(self.delta),
            detect_started=False,
            recent_frame_i=0,
            recent_dt_accum=0.0,
        )
        self._logger.info(
            f"[INIT] state created for camera={camera_id}, hw={frame_hw}"
        )

    def _ensure_camera_initialized(
        self,
        camera_id: str,
        frame_hw: Tuple[int, int],
        ideal_bgr: np.ndarray,
        now_mono: float,
    ) -> None:
        self._init_camera_state_if_needed(camera_id, frame_hw, ideal_bgr, now_mono)
        self._ensure_camera_tracker(camera_id)

    def gamma_rgb01(self, x: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return np.power(x, gamma).astype(np.float32)

    @staticmethod
    def _blur_rgb01(rgb01: np.ndarray, ksize: int = 5) -> np.ndarray:
        if rgb01 is None or ksize <= 1:
            return rgb01
        if (ksize % 2) == 0:
            ksize += 1
        return cv2.GaussianBlur(rgb01, (ksize, ksize), 0)

    def process_frame(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        ideal_bgr: np.ndarray,
        polygons: Optional[List[np.ndarray]] = None,
        now_mono: Optional[float] = None,
        frame_obj: Any = None,
    ) -> dict:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        if ideal_bgr is None:
            raise ValueError("ideal_bgr is None")

        if now_mono is None:
            now_mono = time.monotonic()

        camera_id = str(camera_id)

        # ------------------------------------------------------------------
        # EARLIEST POSSIBLE FAST RESIZE
        # Everything after this point works only in resized resolution.
        # ------------------------------------------------------------------
        cur_bgr, resize_scale = self._resize_bgr_if_needed(
            np.ascontiguousarray(frame_bgr),
            self.max_long_side,
        )
        h, w = cur_bgr.shape[:2]

        if resize_scale != 1.0:
            ideal_bgr_resized = cv2.resize(ideal_bgr, (w, h), interpolation=cv2.INTER_AREA)
            ideal_bgr_resized = np.ascontiguousarray(ideal_bgr_resized)

            polygons_resized = None
            if polygons is not None:
                polygons_resized = [
                    self._resize_polygon(np.asarray(p, dtype=np.float32), resize_scale)
                    for p in polygons
                ]

            if frame_obj is not None:
                raw_humans = getattr(frame_obj, "raw_humans", None)
                raw_human_boxes = getattr(raw_humans, "boxes", None) if raw_humans is not None else None
                if raw_human_boxes is not None:
                    scaled_boxes = [
                        self._scale_box_xyxy(box, resize_scale, (h, w))
                        for box in raw_human_boxes
                    ]
                    raw_humans.boxes = scaled_boxes
        else:
            ideal_bgr_resized = np.ascontiguousarray(ideal_bgr)
            polygons_resized = polygons

        cur_rgb01 = self._bgr_u8_to_rgb01(cur_bgr)

        self._ensure_camera_initialized(
            camera_id=camera_id,
            frame_hw=(h, w),
            ideal_bgr=ideal_bgr_resized,
            now_mono=now_mono,
        )
        st = self.camera_state[camera_id]

        zone_mask_u8 = self._build_zone_mask_u8((h, w), polygons_resized, frame_obj=frame_obj)
        zone_bool = zone_mask_u8 > 0

        alpha_used = self._update_recent(st=st, cur_rgb01=cur_rgb01, now_mono=now_mono)

        if not self._should_detect(st, now_mono):
            return {
                "detected": False,
                "status": False,
                "boxes": np.empty((0, 4), dtype=np.int32),
                "candidate_boxes": np.empty((0, 4), dtype=np.int32),
                "pending_candidate_boxes": np.empty((0, 4), dtype=np.int32),
                "reported_boxes": np.empty((0, 4), dtype=np.int32),
                "instance_masks": [],
                "cd_mask01": None,
                "fused_mask01": None,
                "debug": {
                    "camera_id": camera_id,
                    "alpha_used": alpha_used,
                    "detect_started": st.detect_started,
                    "n_instances": 0,
                    "candidate_len": 0,
                    "reported_len": 0,
                    "skipped_by_delta": True,
                    "resize_scale": resize_scale,
                    "processed_hw": [h, w],
                    "original_hw": list(frame_bgr.shape[:2]),
                },
            }

        recent_rgb01_for_cd = (
            (1.0 - self.min_empty_weight) * st.recent_rgb01
            + self.min_empty_weight * st.empty_rgb01
        )

        empty_rgb01 = self.gamma_rgb01(st.empty_rgb01, 0.8)
        empty_rgb01 = self._blur_rgb01(empty_rgb01, ksize=3)

        recent_rgb01_for_cd = self.gamma_rgb01(recent_rgb01_for_cd, 0.8)
        blur_size = max(3, (recent_rgb01_for_cd.shape[1] // 40) | 1)
        recent_blur = self._blur_rgb01(recent_rgb01_for_cd, ksize=blur_size)
        recent_rgb01_for_cd = (
            (1.0 - self.min_empty_weight) * recent_blur
            + self.min_empty_weight * st.empty_rgb01
        )

        cur_rgb01_proc = self.gamma_rgb01(cur_rgb01, 0.8)
        cur_rgb01_proc = self._blur_rgb01(cur_rgb01_proc, ksize=3)

        cd_mask01 = self._infer_cd_mask01(
            empty_rgb01,
            recent_rgb01_for_cd,
            cur_rgb01_proc,
            camera_id,
        )

        if not st.detect_started:
            st.detect_started = True

        fused01 = cd_mask01.copy()
        fused01 *= zone_bool.astype(np.float32)

        self._logger.info(
            f"[FUSE] camera={camera_id} cd_sum={cd_mask01.sum():.1f}, "
            f"fused_sum={fused01.sum():.1f}, resize_scale={resize_scale:.4f}, hw={(h, w)}"
        )

        inst_masks = self._mask01_to_instances(fused01)
        current_bboxes = [self._get_bbox(m) for m in inst_masks]

        tracker = self.tracker_by_camera[camera_id]
        candidate_boxes, reported_boxes = tracker.update(
            now=now_mono,
            current_bboxes=current_bboxes,
            current_masks=inst_masks,
        )

        save_rt_panel(
            camera_id=camera_id,
            iter_idx=self._rt_iter_by_camera.get(camera_id, 0) + 1,
            empty_rgb01=empty_rgb01,
            recent_rgb01=recent_rgb01_for_cd,
            cur_rgb01=cur_rgb01,
            cd_mask01=cd_mask01,
            candidate_boxes=candidate_boxes,
            reported_boxes=reported_boxes,
            logger=self._logger,
        )
        self._rt_iter_by_camera[camera_id] = self._rt_iter_by_camera.get(camera_id, 0) + 1

        self._logger.info(
            f"[TRACK] camera={camera_id} "
            f"current={len(current_bboxes)} candidate_now={len(candidate_boxes)} "
            f"reported_now={len(reported_boxes)} stationary_time_sec={self.stationary_time_sec}"
        )

        return {
            "detected": True,
            "status": (len(candidate_boxes) > 0 or len(reported_boxes) > 0),
            "candidate_boxes": candidate_boxes,
            "reported_boxes": reported_boxes,
            "instance_masks": inst_masks,
            "cd_mask01": cd_mask01,
            "fused_mask01": fused01,
            "debug": {
                "camera_id": camera_id,
                "alpha_used": alpha_used,
                "detect_started": st.detect_started,
                "n_instances": len(inst_masks),
                "current_frame_boxes_len": len(current_bboxes),
                "candidate_len": len(candidate_boxes),
                "reported_len": len(reported_boxes),
                "skipped_by_delta": False,
                "resize_scale": resize_scale,
                "processed_hw": [h, w],
                "original_hw": list(frame_bgr.shape[:2]),
            },
        }