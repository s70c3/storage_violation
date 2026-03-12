from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Candidate:
    cand_id: int
    first_seen: float
    last_seen: float
    hits: int
    mask: np.ndarray
    bbox: np.ndarray
    misses: int = 0


class CandidateQueueManager:
    """
    Keeps and updates persistent candidates between detect ticks.
    Matching is greedy with gating.
    """

    def __init__(
        self,
        threshold_hits: int = 3,
        threshold_time_sec: float = 0.0,
        expiration_time_sec: float = 60.0,
        miss_tolerance: int = 2,
        reset_on_miss: bool = True,
        pad_factor: float = 0.2,
        min_bbox_iou_gate: float = 0.3,
        min_mask_iou: float = 0.6,
        w_bbox: float = 0.5,
        w_centroid: float = 0.3,
        w_mask: float = 0.2,
        logger: logging.Logger | None = None,
    ):
        self.threshold_hits = int(threshold_hits)
        self.threshold_time_sec = float(threshold_time_sec)
        self.expiration_time_sec = float(expiration_time_sec)

        self.miss_tolerance = int(miss_tolerance)
        self.reset_on_miss = bool(reset_on_miss)

        self.pad_factor = float(pad_factor)
        self.min_bbox_iou_gate = float(min_bbox_iou_gate)
        self.min_mask_iou = float(min_mask_iou)
        self.w_bbox = float(w_bbox)
        self.w_centroid = float(w_centroid)
        self.w_mask = float(w_mask)

        self._candidates: Dict[int, Candidate] = {}
        self._next_id: int = 1

        self.logger = logger or logging.getLogger("StorageViolationQueue")

    def snapshot_ids(self) -> List[int]:
        return sorted(self._candidates.keys())

    def snapshot(self) -> List[Candidate]:
        return [self._candidates[cid] for cid in self.snapshot_ids()]

    def expire(self, now: float) -> None:
        if self.expiration_time_sec <= 0:
            return
        dead = [
            cid for cid, c in self._candidates.items()
            if (now - c.last_seen) > self.expiration_time_sec
        ]
        for cid in dead:
            del self._candidates[cid]

    @staticmethod
    def _bbox_iou(b1, b2) -> float:
        b1 = np.asarray(b1, dtype=np.float32)
        b2 = np.asarray(b2, dtype=np.float32)
        inter_wh = np.maximum(0.0, np.minimum(b1[2:], b2[2:]) - np.maximum(b1[:2], b2[:2]))
        inter = float(inter_wh[0] * inter_wh[1])
        a1 = float(np.maximum(0.0, b1[2] - b1[0]) * np.maximum(0.0, b1[3] - b1[1]))
        a2 = float(np.maximum(0.0, b2[2] - b2[0]) * np.maximum(0.0, b2[3] - b2[1]))
        union = a1 + a2 - inter
        return inter / union if union > 1e-9 else 0.0

    @staticmethod
    def _mask_iou(m1, m2) -> float:
        inter = float(np.logical_and(m1, m2).sum())
        union = float(np.logical_or(m1, m2).sum())
        return inter / union if union > 1e-9 else 0.0

    @staticmethod
    def _centroid_norm_dist(b1, b2) -> float:
        b1 = np.asarray(b1, dtype=np.float32)
        b2 = np.asarray(b2, dtype=np.float32)
        c1 = (b1[:2] + b1[2:]) * 0.5
        c2 = (b2[:2] + b2[2:]) * 0.5
        diag = float(np.linalg.norm(b1[2:] - b1[:2]) + 1e-6)
        return float(np.tanh(np.linalg.norm(c1 - c2) / diag))

    def _associate(
        self,
        ref_masks: List[np.ndarray],
        ref_bboxes: List[np.ndarray],
        cur_masks: List[np.ndarray],
        cur_bboxes: List[np.ndarray],
    ) -> List[Tuple[int, int]]:
        if not ref_masks or not cur_masks:
            return []

        n_ref, n_cur = len(ref_masks), len(cur_masks)
        triples: List[Tuple[float, int, int]] = []

        for i_ref, rb in enumerate(ref_bboxes):
            rb = np.asarray(rb, dtype=np.float32).reshape(-1)
            diag = float(np.hypot(rb[2] - rb[0], rb[3] - rb[1]))
            pad = self.pad_factor * diag
            qx1, qy1, qx2, qy2 = float(rb[0] - pad), float(rb[1] - pad), float(rb[2] + pad), float(rb[3] + pad)

            for i_cur, cb in enumerate(cur_bboxes):
                cb = np.asarray(cb, dtype=np.float32).reshape(-1)

                if cb[2] < qx1 or cb[0] > qx2 or cb[3] < qy1 or cb[1] > qy2:
                    continue

                biou = self._bbox_iou(rb, cb)
                if biou < self.min_bbox_iou_gate:
                    continue

                miou = self._mask_iou(ref_masks[i_ref], cur_masks[i_cur])
                if miou < self.min_mask_iou:
                    continue

                cd = self._centroid_norm_dist(rb, cb)

                cost = (
                    self.w_bbox * (1.0 - biou)
                    + self.w_centroid * cd
                    + self.w_mask * (1.0 - miou)
                )
                if np.isfinite(cost):
                    triples.append((float(cost), int(i_ref), int(i_cur)))

        if not triples:
            return []

        triples.sort(key=lambda x: x[0])

        used_ref = np.zeros(n_ref, dtype=bool)
        used_cur = np.zeros(n_cur, dtype=bool)
        pairs: List[Tuple[int, int]] = []

        for _, r, c in triples:
            if used_ref[r] or used_cur[c]:
                continue
            used_ref[r] = True
            used_cur[c] = True
            pairs.append((r, c))

        return pairs

    def step(
        self,
        now: float,
        current_masks: List[np.ndarray],
        current_bboxes: List[np.ndarray],
    ) -> Tuple[List[Candidate], List[Candidate]]:
        n = len(current_masks)
        if len(current_bboxes) != n:
            raise ValueError("current_masks/current_bboxes must have same length")

        snap_ids = self.snapshot_ids()
        snapshot_before = [self._candidates[cid] for cid in snap_ids]

        ref_masks = [c.mask for c in snapshot_before]
        ref_bboxes = [c.bbox for c in snapshot_before]
        pairs = self._associate(ref_masks, ref_bboxes, current_masks, current_bboxes)

        matched_cur = {c for _, c in pairs}
        unmatched_cur = [i for i in range(n) if i not in matched_cur]

        matched_cand_ids: set[int] = set()

        for ref_i, cur_i in pairs:
            if ref_i < 0 or ref_i >= len(snap_ids):
                continue
            cand_id = snap_ids[ref_i]
            cand = self._candidates.get(cand_id)
            if cand is None:
                continue

            matched_cand_ids.add(cand_id)

            if self.reset_on_miss and cand.misses > self.miss_tolerance:
                cand.hits = 0
                cand.first_seen = now

            cand.misses = 0
            cand.last_seen = now
            cand.hits += 1
            cand.mask = current_masks[cur_i]
            cand.bbox = current_bboxes[cur_i]

        for cand_id, cand in self._candidates.items():
            if cand_id not in matched_cand_ids:
                cand.misses += 1

        for cur_i in unmatched_cur:
            cid = self._next_id
            self._next_id += 1
            self._candidates[cid] = Candidate(
                cand_id=cid,
                first_seen=now,
                last_seen=now,
                hits=1,
                mask=current_masks[cur_i],
                bbox=current_bboxes[cur_i],
                misses=0,
            )

        self.expire(now)
        ready_after = self.ready(now)
        return snapshot_before, ready_after

    def ready(self, now: float) -> List[Candidate]:
        out: List[Candidate] = []
        for c in self._candidates.values():
            if c.misses != 0:
                continue

            stable = False
            if self.threshold_hits > 0 and c.hits >= self.threshold_hits:
                stable = True
            if self.threshold_time_sec > 0 and (now - c.first_seen) >= self.threshold_time_sec:
                stable = True

            if stable:
                out.append(c)
        return out
