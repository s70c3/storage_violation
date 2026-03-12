from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class IdealImageStorage:
    def __init__(self):
        self._images: Dict[str, np.ndarray] = {}

    def save(self, camera_id: str, image_bgr: np.ndarray) -> None:
        self._images[str(camera_id)] = image_bgr.copy()

    def load(self, camera_id: str) -> Optional[np.ndarray]:
        image = self._images.get(str(camera_id))
        if image is None:
            return None
        return image.copy()

    def exists(self, camera_id: str) -> bool:
        return str(camera_id) in self._images

    def delete(self, camera_id: str) -> bool:
        camera_id = str(camera_id)
        if camera_id in self._images:
            del self._images[camera_id]
            return True
        return False

    def clear(self) -> None:
        self._images.clear()

    def list_camera_ids(self) -> list[str]:
        return sorted(self._images.keys())