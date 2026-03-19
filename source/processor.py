from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch

from .queue_manager import CandidateQueueManager
from .segmentator import SemanticSegmentatorProtocol


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
    Independent per-frame processor.

    Main public method:
        process_frame(camera_id, frame_bgr, ideal_bgr, polygons=None, frame_obj=None)

    Returns:
        - candidate_boxes: red boxes, current candidates from queue snapshot
        - reported_boxes: green boxes, confirmed abandoned objects
        - boxes: alias of reported_boxes for backward compatibility
    """

    def __init__(
        self,
        cd_segmentator: SemanticSegmentatorProtocol,
        delta: int = 0,

        ema_tau_sec: float = 5.0,
        ema_min_alpha: float = 0.02,
        ema_max_alpha: float = 0.12,

        min_empty_weight: float = 0.20,

        thr: float = 0.5,
        min_side: int = 100,
        morph_ksize: int = 3,

        threshold_hits: int = 10,
        threshold_time_sec: float = 5.0,
        expiration_time_sec: float = 30.0,
        drop_reported: bool = False,

        pad_factor: float = 0.2,
        min_bbox_iou_gate: float = 0.3,
        min_mask_iou: float = 0.6,
        w_bbox: float = 0.5,
        w_centroid: float = 0.3,
        w_mask: float = 0.2,

        tile_split_long_side: int = 5000,
        tile_overlap: int = 64,

        recent_update_every: int = 10,

        logger: logging.Logger | None = None,
    ):
        self.cd_segmentator = cd_segmentator
        self.delta = int(delta)

        self.camera_state: Dict[str, CameraEMAState] = {}
        self.queue_by_camera: Dict[str, CandidateQueueManager] = {}

        self.ema_tau_sec = float(ema_tau_sec)
        self.ema_min_alpha = float(ema_min_alpha)
        self.ema_max_alpha = float(ema_max_alpha)

        self.min_empty_weight = float(min_empty_weight)

        self.thr = float(thr)
        self.min_side = int(min_side)
        self.morph_ksize = int(morph_ksize)

        self.threshold_hits = int(threshold_hits)
        self.threshold_time_sec = float(threshold_time_sec)
        self.expiration_time_sec = float(expiration_time_sec)
        self.drop_reported = bool(drop_reported)

        self.pad_factor = float(pad_factor)
        self.min_bbox_iou_gate = float(min_bbox_iou_gate)
        self.min_mask_iou = float(min_mask_iou)
        self.w_bbox = float(w_bbox)
        self.w_centroid = float(w_centroid)
        self.w_mask = float(w_mask)

        self.tile_split_long_side = int(tile_split_long_side)
        self.tile_overlap = int(max(0, tile_overlap))

        self.recent_update_every = max(1, int(recent_update_every))

        self._logger = logger or logging.getLogger("StorageViolationProcessor")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self._cd_mean = np.asarray([0.485, 0.456, 0.406], np.float32)
        self._cd_std = np.asarray([0.229, 0.224, 0.225], np.float32)

    def update_runtime_params(
        self,
        threshold_hits: int | None = None,
        threshold_time_sec: float | None = None,
        min_side: int | None = None,
    ):
        if threshold_hits is not None:
            self.threshold_hits = int(threshold_hits)

        if threshold_time_sec is not None:
            self.threshold_time_sec = float(threshold_time_sec)

        if min_side is not None:
            self.min_side = int(min_side)

        self._logger.info(
            f"[PARAM_UPDATE] threshold_hits={self.threshold_hits} "
            f"threshold_time_sec={self.threshold_time_sec} "
            f"min_side={self.min_side}"
        )

        for camera_id in list(self.queue_by_camera.keys()):
            self.queue_by_camera[camera_id] = self._make_queue_manager()
            self._logger.info(f"[PARAM_UPDATE] queue reset for camera={camera_id}")

    def load(self) -> None:
        self.cd_segmentator.load()

    def close(self) -> None:
        if self.cd_segmentator is not None:
            self.cd_segmentator.close()
        self.camera_state.clear()
        self.queue_by_camera.clear()

    def reset_camera(self, camera_id: str) -> None:
        camera_id = str(camera_id)
        self.camera_state.pop(camera_id, None)
        self.queue_by_camera.pop(camera_id, None)

    def _make_queue_manager(self) -> CandidateQueueManager:
        return CandidateQueueManager(
            threshold_hits=self.threshold_hits,
            threshold_time_sec=self.threshold_time_sec,
            expiration_time_sec=self.expiration_time_sec,
            pad_factor=self.pad_factor,
            min_bbox_iou_gate=self.min_bbox_iou_gate,
            min_mask_iou=self.min_mask_iou,
            w_bbox=self.w_bbox,
            w_centroid=self.w_centroid,
            w_mask=self.w_mask,
            logger=self._logger,
        )

    def _ensure_camera_queue(self, camera_id: str) -> None:
        if camera_id not in self.queue_by_camera:
            self.queue_by_camera[camera_id] = self._make_queue_manager()
            self._logger.info(f"[INIT] queue created for camera={camera_id}")

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
    def _fit_rgb01_to_hw(rgb01: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        th, tw = target_hw
        return cv2.resize(rgb01, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)

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

    @staticmethod
    def _extract_candidate_boxes(candidates: List[Any]) -> np.ndarray:
        if not candidates:
            return np.empty((0, 4), dtype=np.int32)
        boxes = []
        for c in candidates:
            bbox = getattr(c, "bbox", None)
            if bbox is None:
                continue
            boxes.append(np.asarray(bbox, dtype=np.int32))
        if not boxes:
            return np.empty((0, 4), dtype=np.int32)
        return np.stack(boxes, axis=0)

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

        h, w = cur_rgb01.shape[:2]
        long_side = max(h, w)
        split = long_side > self.tile_split_long_side
        ov = int(self.tile_overlap)

        def infer_tile(e_tile: np.ndarray, r_tile: np.ndarray, c_tile: np.ndarray) -> np.ndarray:
            x = self._build_cd_tensor_1x9(e_tile, r_tile, c_tile)

            pred_fn = getattr(self.cd_segmentator, "predict_semantic01", None)
            if callable(pred_fn):
                sem01 = pred_fn(x, {"camera_id": camera_id})
                sem01 = np.asarray(sem01, dtype=np.float32)
                exp_hw = (c_tile.shape[0], c_tile.shape[1])
                if sem01.shape != exp_hw:
                    self._logger.warning(
                        f"[CD_TILE] predict_semantic01 bad shape={sem01.shape}, expected={exp_hw}"
                    )
                    return np.zeros(exp_hw, np.float32)
                return sem01

            cd_res = self.cd_segmentator([x], {"camera_id": camera_id})
            if not cd_res:
                return np.zeros((c_tile.shape[0], c_tile.shape[1]), np.float32)

            bm = getattr(cd_res[0], "binary_masks", None)
            masks = [] if (bm is None or bm.masks is None) else (bm.masks or [])
            if not masks:
                return np.zeros((c_tile.shape[0], c_tile.shape[1]), np.float32)

            sem = np.zeros((c_tile.shape[0], c_tile.shape[1]), dtype=bool)
            for m in masks:
                sem |= np.asarray(m, dtype=bool)

            return sem.astype(np.float32, copy=False)

        if not split:
            return infer_tile(empty_rgb01, recent_rgb01, cur_rgb01)

        hm = h // 2
        wm = w // 2
        base_tiles = [
            (0, hm, 0, wm),
            (0, hm, wm, w),
            (hm, h, 0, wm),
            (hm, h, wm, w),
        ]

        out = np.zeros((h, w), np.float32)

        for (y0, y1, x0, x1) in base_tiles:
            ey0 = max(0, y0 - ov)
            ey1 = min(h, y1 + ov)
            ex0 = max(0, x0 - ov)
            ex1 = min(w, x1 + ov)

            pred = infer_tile(
                empty_rgb01[ey0:ey1, ex0:ex1],
                recent_rgb01[ey0:ey1, ex0:ex1],
                cur_rgb01[ey0:ey1, ex0:ex1],
            )

            exp_hw = (ey1 - ey0, ex1 - ex0)
            if pred.shape != exp_hw:
                self._logger.warning(
                    f"[CD_TILE] pred shape mismatch {pred.shape} vs {exp_hw}; using zeros"
                )
                pred = np.zeros(exp_hw, np.float32)

            out[ey0:ey1, ex0:ex1] = np.maximum(out[ey0:ey1, ex0:ex1], pred)

        return out

    def _init_camera_state_if_needed(
        self,
        camera_id: str,
        frame_hw: Tuple[int, int],
        ideal_bgr: np.ndarray,
        now_mono: float,
    ) -> None:
        if camera_id in self.camera_state:
            return

        h, w = frame_hw
        empty_rgb01_raw = self._bgr_u8_to_rgb01(ideal_bgr)
        if empty_rgb01_raw.shape[:2] != (h, w):
            self._logger.warning(
                f"[INIT] camera={camera_id} ideal size={empty_rgb01_raw.shape[:2]} != frame size={(h, w)}; using resize"
            )

        empty_rgb01 = self._fit_rgb01_to_hw(empty_rgb01_raw, (h, w))

        self.camera_state[camera_id] = CameraEMAState(
            empty_rgb01=empty_rgb01,
            recent_rgb01=empty_rgb01.copy(),
            last_ts_mono=now_mono,
            last_detect_mono=now_mono - float(self.delta),
            detect_started=False,
            recent_frame_i=0,
            recent_dt_accum=0.0,
        )
        self._logger.info(f"[INIT] state created for camera={camera_id}")

    def _ensure_camera_initialized(
        self,
        camera_id: str,
        frame_hw: Tuple[int, int],
        ideal_bgr: np.ndarray,
        now_mono: float,
    ) -> None:
        self._init_camera_state_if_needed(camera_id, frame_hw, ideal_bgr, now_mono)
        self._ensure_camera_queue(camera_id)

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
        cur_bgr = np.ascontiguousarray(frame_bgr)
        h, w = cur_bgr.shape[:2]
        cur_rgb01 = self._bgr_u8_to_rgb01(cur_bgr)

        self._ensure_camera_initialized(
            camera_id=camera_id,
            frame_hw=(h, w),
            ideal_bgr=ideal_bgr,
            now_mono=now_mono,
        )
        st = self.camera_state[camera_id]

        zone_mask_u8 = self._build_zone_mask_u8((h, w), polygons, frame_obj=frame_obj)
        zone_bool = zone_mask_u8 > 0

        alpha_used = self._update_recent(st=st, cur_rgb01=cur_rgb01, now_mono=now_mono)

        if not self._should_detect(st, now_mono):
            return {
                "detected": False,
                "status": False,
                "boxes": np.empty((0, 4), dtype=np.int32),
                "candidate_boxes": np.empty((0, 4), dtype=np.int32),
                "reported_boxes": np.empty((0, 4), dtype=np.int32),
                "instance_masks": [],
                "cd_mask01": None,
                "fused_mask01": None,
                "debug": {
                    "camera_id": camera_id,
                    "alpha_used": alpha_used,
                    "detect_started": st.detect_started,
                    "n_instances": 0,
                    "snapshot_len": 0,
                    "ready_len": 0,
                    "skipped_by_delta": True,
                },
            }

        recent_rgb01_for_cd = (
            (1.0 - self.min_empty_weight) * st.recent_rgb01
            + self.min_empty_weight * st.empty_rgb01
        )

        empty_rgb01 = self.gamma_rgb01(st.empty_rgb01, 0.8)
        empty_rgb01 = self._blur_rgb01(empty_rgb01, ksize=13)

        recent_rgb01_for_cd = self.gamma_rgb01(recent_rgb01_for_cd, 0.8)
        recent_blur = self._blur_rgb01(recent_rgb01_for_cd, ksize=51)
        recent_rgb01_for_cd = (
            (1.0 - self.min_empty_weight) * recent_blur
            + self.min_empty_weight * st.empty_rgb01
        )

        cur_rgb01_proc = self.gamma_rgb01(cur_rgb01, 0.8)
        cur_rgb01_proc = self._blur_rgb01(cur_rgb01_proc, ksize=13)

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
            f"[FUSE] camera={camera_id} cd_sum={cd_mask01.sum():.1f}, fused_sum={fused01.sum():.1f}"
        )

        inst_masks = self._mask01_to_instances(fused01)

        cq = self.queue_by_camera[camera_id]
        bboxes = [self._get_bbox(m) for m in inst_masks]
        snap, ready = cq.step(now=now_mono, current_masks=inst_masks, current_bboxes=bboxes)

        candidate_boxes = self._extract_candidate_boxes(snap)
        reported_boxes = self._extract_candidate_boxes(ready)

        self._logger.info(
            f"[QUEUE] camera={camera_id} snapshot={len(snap)}, ready={len(ready)}, threshold_hits={self.threshold_hits}"
        )

        if len(ready) > 0 and hasattr(cq, "on_report"):
            cq.on_report([c.cand_id for c in ready])

        return {
            "detected": True,
            "status": len(ready) > 0,
            "boxes": reported_boxes,          # backward compatibility
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
                "snapshot_len": len(snap),
                "ready_len": len(ready),
                "skipped_by_delta": False,
            },
        }