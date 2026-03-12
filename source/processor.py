from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    detect_started: bool = False
    stable_u8: Optional[np.ndarray] = None
    stable_last_seen_mono: float = 0.0
    recent_frame_i: int = 0
    recent_dt_accum: float = 0.0


class StorageViolationFrameProcessor:
    """
    Independent per-frame processor.

    Main public method:
        process_frame(camera_id, frame_bgr, ideal_bgr, polygons=None)
    """

    def __init__(
        self,
        cd_segmentator: SemanticSegmentatorProtocol,
        ema_tau_sec: float = 5.0,
        pre_detect_alpha: float = 0.03,
        thr: float = 0.5,
        min_side: int = 100,
        threshold_hits: int = 10,
        threshold_time_sec: float = 5.0,
        expiration_time_sec: float = 30.0,
        stable_ttl_sec: float = 60.0,
        inpaint_radius: int = 3,
        tile_split_long_side: int = 1000,
        tile_overlap: int = 64,
        recent_update_every: int = 10,
        min_empty_weight: float = 0.10,
        logger: logging.Logger | None = None,
    ):
        self.cd_segmentator = cd_segmentator

        self.camera_state: Dict[str, CameraEMAState] = {}
        self.queue_by_camera: Dict[str, CandidateQueueManager] = {}

        # Public tuning params
        self.ema_tau_sec = float(ema_tau_sec)
        self.pre_detect_alpha = float(pre_detect_alpha)
        self.thr = float(thr)
        self.min_side = int(min_side)

        self.threshold_hits = int(threshold_hits)
        self.threshold_time_sec = float(threshold_time_sec)
        self.expiration_time_sec = float(expiration_time_sec)

        self.stable_ttl_sec = float(stable_ttl_sec)
        self.inpaint_radius = int(inpaint_radius)

        self.tile_split_long_side = int(tile_split_long_side)
        self.tile_overlap = int(max(0, tile_overlap))
        self.recent_update_every = int(recent_update_every)
        self.min_empty_weight = float(min_empty_weight)

        self._logger = logger or logging.getLogger("StorageViolationProcessor")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Private/internal constants
        self._ema_min_alpha = 0.02
        self._ema_max_alpha = 0.12

        self._pre_detect_alpha_min = 0.01
        self._pre_detect_alpha_max = 0.05

        self._morph_ksize = 0

        self._pad_factor = 0.2
        self._min_bbox_iou_gate = 0.3
        self._min_mask_iou = 0.6
        self._w_bbox = 0.5
        self._w_centroid = 0.3
        self._w_mask = 0.2

        self._stable_dilate_ksize = 9

        self._cd_mean = np.asarray([0.485, 0.456, 0.406], np.float32)
        self._cd_std = np.asarray([0.229, 0.224, 0.225], np.float32)

    def update_runtime_params(
            self,
            threshold_hits: int | None = None,
            threshold_time_sec: float | None = None,
            min_side: int | None = None,
    ):
        """
        Update runtime parameters affecting candidate filtering.
        Existing camera queues will be recreated.
        """

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

        # recreate queue managers for cameras
        for camera_id in list(self.queue_by_camera.keys()):
            self.queue_by_camera[camera_id] = CandidateQueueManager(
                threshold_hits=self.threshold_hits,
                threshold_time_sec=self.threshold_time_sec,
                expiration_time_sec=self.expiration_time_sec,
                drop_reported=self.drop_reported,
                pad_factor=self.pad_factor,
                min_bbox_iou_gate=self.min_bbox_iou_gate,
                min_mask_iou=self.min_mask_iou,
                w_bbox=self.w_bbox,
                w_centroid=self.w_centroid,
                w_mask=self.w_mask,
                logger=self._logger,
            )

            self._logger.info(f"[PARAM_UPDATE] queue reset for camera={camera_id}")

    def load(self) -> None:
        self.cd_segmentator.load()

    def close(self) -> None:
        if self.cd_segmentator is not None:
            self.cd_segmentator.close()
        self.camera_state.clear()
        self.queue_by_camera.clear()

    def reset_camera(self, camera_id: str) -> None:
        self.camera_state.pop(str(camera_id), None)
        self.queue_by_camera.pop(str(camera_id), None)


    def _ema_alpha(self, dt: float) -> float:
        if self.ema_tau_sec <= 1e-6:
            a = 1.0
        else:
            a = 1.0 - math.exp(-max(0.0, dt) / self.ema_tau_sec)
        return float(np.clip(a, self._ema_min_alpha, self._ema_max_alpha))

    @staticmethod
    def _bgr_u8_to_rgb01(bgr_u8: np.ndarray) -> np.ndarray:
        return bgr_u8[..., ::-1].astype(np.float32) / 255.0

    @staticmethod
    def _rgb01_to_bgr_u8(rgb01: np.ndarray) -> np.ndarray:
        rgb01 = np.clip(rgb01, 0.0, 1.0)
        return (rgb01[..., ::-1] * 255.0).astype(np.uint8)

    @staticmethod
    def _fit_rgb01_to_hw(rgb01: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        th, tw = target_hw
        return cv2.resize(rgb01, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def _build_zone_mask_u8(
        self,
        image_hw: Tuple[int, int],
        polygons: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        h, w = image_hw
        zone_mask = np.zeros((h, w), dtype=np.uint8)

        if polygons:
            for polygon in polygons:
                pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zone_mask, [pts], 255)
        else:
            zone_mask[:, :] = 255

        return zone_mask

    def _mask01_to_instances(self, mask01: np.ndarray) -> List[np.ndarray]:
        if mask01 is None:
            return []

        m = (mask01 >= self.thr).astype(np.uint8) * 255

        if self._morph_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._morph_ksize, self._morph_ksize))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

        m = np.ascontiguousarray(m)
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
    def _masks_to_u8_union(
        masks: List[np.ndarray],
        hw: Tuple[int, int],
        dilate_ksize: int = 9,
    ) -> np.ndarray:
        h, w = hw
        if not masks:
            return np.zeros((h, w), np.uint8)

        u = np.zeros((h, w), np.uint8)
        for m in masks:
            u[m] = 255

        if dilate_ksize and dilate_ksize > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
            u = cv2.dilate(u, k, iterations=1)

        return u

    def _inpaint_recent_zone(
        self,
        recent_rgb01: np.ndarray,
        hole_u8: np.ndarray,
        zone_u8: np.ndarray,
    ) -> np.ndarray:
        if hole_u8 is None:
            return recent_rgb01
        if zone_u8 is None:
            zone_u8 = np.ones(hole_u8.shape[:2], np.uint8) * 255

        hole = hole_u8.copy()
        hole &= zone_u8
        if int(hole.max()) == 0:
            return recent_rgb01

        bgr = self._rgb01_to_bgr_u8(recent_rgb01)
        out = cv2.inpaint(bgr, hole, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        return self._bgr_u8_to_rgb01(out)

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

    def _infer_cd_mask01(
        self,
        empty_rgb01: np.ndarray,
        recent_rgb01: np.ndarray,
        cur_rgb01: np.ndarray,
        camera_id: str,
    ) -> np.ndarray:
        h, w = cur_rgb01.shape[:2]
        long_side = max(h, w)
        split = long_side > self.tile_split_long_side
        ov = int(self.tile_overlap)

        def infer_tile(e_tile, r_tile, c_tile) -> np.ndarray:
            x = self._build_cd_tensor_1x9(e_tile, r_tile, c_tile)
            sem01 = self.cd_segmentator.predict_semantic01(x, {"camera_id": camera_id})
            sem01 = np.asarray(sem01, dtype=np.float32)

            exp_hw = (c_tile.shape[0], c_tile.shape[1])
            if sem01.shape != exp_hw:
                self._logger.warning(
                    f"[CD_TILE] bad shape={sem01.shape}, expected={exp_hw}; return zeros"
                )
                return np.zeros(exp_hw, np.float32)
            return sem01

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
                pred = np.zeros(exp_hw, np.float32)

            out[ey0:ey1, ex0:ex1] = np.maximum(out[ey0:ey1, ex0:ex1], pred)

        return out

    def _update_recent(
        self,
        st: CameraEMAState,
        cur_rgb01: np.ndarray,
        now_mono: float,
        zone_mask_u8: np.ndarray,
    ) -> float:
        dt = now_mono - st.last_ts_mono
        st.last_ts_mono = now_mono

        st.recent_frame_i += 1
        st.recent_dt_accum += max(0.0, float(dt))

        if (st.recent_frame_i % self.recent_update_every) != 0:
            return 0.0

        dt_eff = st.recent_dt_accum
        st.recent_dt_accum = 0.0

        if not st.detect_started:
            a = float(np.clip(
                self.pre_detect_alpha,
                self._pre_detect_alpha_min,
                self._pre_detect_alpha_max,
            ))
            st.recent_rgb01 = (1.0 - a) * st.empty_rgb01 + a * cur_rgb01
            return a

        a = self._ema_alpha(dt_eff)
        st.recent_rgb01 = (1.0 - a) * st.recent_rgb01 + a * cur_rgb01

        if st.stable_u8 is not None and (now_mono - st.stable_last_seen_mono) < self.stable_ttl_sec:
            st.recent_rgb01 = self._inpaint_recent_zone(st.recent_rgb01, st.stable_u8, zone_mask_u8)
        elif st.stable_u8 is not None:
            st.stable_u8 = None

        return float(a)

    def _queue_step(self, camera_id: str, now_mono: float, inst_masks: List[np.ndarray]):
        cq = self.queue_by_camera[camera_id]
        bboxes = [self._get_bbox(m) for m in inst_masks]
        snap, ready = cq.step(now=now_mono, current_masks=inst_masks, current_bboxes=bboxes)
        return snap, ready, bboxes

    def _init_camera_if_needed(
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
                f"[INIT] camera={camera_id} ideal size={empty_rgb01_raw.shape[:2]} "
                f"!= frame size={(h, w)}; using center crop/pad"
            )

        empty_rgb01 = self._fit_rgb01_to_hw(empty_rgb01_raw, (h, w))

        self.camera_state[camera_id] = CameraEMAState(
            empty_rgb01=empty_rgb01,
            recent_rgb01=empty_rgb01.copy(),
            last_ts_mono=now_mono,
            detect_started=False,
            stable_u8=None,
            stable_last_seen_mono=0.0,
            recent_frame_i=0,
            recent_dt_accum=0.0,
        )

        self.queue_by_camera[camera_id] = CandidateQueueManager(
            threshold_hits=self.threshold_hits,
            threshold_time_sec=self.threshold_time_sec,
            expiration_time_sec=self.expiration_time_sec,
            pad_factor=self._pad_factor,
            min_bbox_iou_gate=self._min_bbox_iou_gate,
            min_mask_iou=self._min_mask_iou,
            w_bbox=self._w_bbox,
            w_centroid=self._w_centroid,
            w_mask=self._w_mask,
            logger=self._logger,
        )

    def process_frame(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        ideal_bgr: np.ndarray,
        polygons: Optional[List[np.ndarray]] = None,
        now_mono: Optional[float] = None,
    ) -> dict:
        if now_mono is None:
            now_mono = time.monotonic()

        camera_id = str(camera_id)
        cur_bgr = frame_bgr
        h, w = cur_bgr.shape[:2]
        cur_rgb01 = self._bgr_u8_to_rgb01(cur_bgr)

        self._init_camera_if_needed(camera_id, (h, w), ideal_bgr, now_mono)
        st = self.camera_state[camera_id]

        zone_mask_u8 = self._build_zone_mask_u8((h, w), polygons)
        zone_bool = zone_mask_u8 > 0

        alpha_used = self._update_recent(st, cur_rgb01, now_mono, zone_mask_u8)

        recent_rgb01 = (
            (1.0 - self.min_empty_weight) * st.recent_rgb01
            + self.min_empty_weight * st.empty_rgb01
        )

        cd_mask01 = self._infer_cd_mask01(st.empty_rgb01, recent_rgb01, cur_rgb01, camera_id)

        if not st.detect_started:
            st.detect_started = True

        fused01 = cd_mask01.copy()
        fused01 *= zone_bool

        inst_masks = self._mask01_to_instances(fused01)

        snap, ready, _ = self._queue_step(camera_id=camera_id, now_mono=now_mono, inst_masks=inst_masks)

        stable_refreshed = False
        if ready:
            stable_masks = [c.mask for c in ready]
            stable_u8 = self._masks_to_u8_union(
                stable_masks,
                (h, w),
                dilate_ksize=self._stable_dilate_ksize,
            )
            stable_u8 &= zone_mask_u8

            st.stable_u8 = stable_u8
            st.stable_last_seen_mono = now_mono
            st.recent_rgb01 = self._inpaint_recent_zone(st.recent_rgb01, st.stable_u8, zone_mask_u8)
            stable_refreshed = True

        if not ready:
            return {
                "detected": True,
                "status": False,
                "boxes": np.empty((0, 4), dtype=np.int32),
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
                    "stable_refreshed": stable_refreshed,
                },
            }

        out_boxes = np.array([c.bbox for c in ready], dtype=np.int32)
        cq = self.queue_by_camera[camera_id]

        return {
            "detected": True,
            "status": True,
            "boxes": out_boxes,
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
                "stable_refreshed": stable_refreshed,
            },
        }