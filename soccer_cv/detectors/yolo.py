from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ultralytics import YOLO

from soccer_cv.types import Detection, clamp_xyxy

COCO_PERSON = "person"
COCO_SPORTS_BALL = "sports ball"


def _resolve_device(device: str) -> Optional[str]:
    device = (device or "auto").strip().lower()
    if device == "auto":
        # Let Ultralytics decide the best device if we pass None.
        return None
    return device


def _half_supported(device: Optional[str]) -> bool:
    # FP16 is reliably supported on CUDA. On CPU/MPS, many kernels are missing.
    if device is None:
        return False
    return str(device).lower().startswith("cuda")


@dataclass
class YoloDetector:
    model_path: str
    device: str = "auto"
    half: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.model = YOLO(self.model_path)
        self._device = _resolve_device(self.device)
        self._use_half = bool(self.half) and _half_supported(self._device)

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        classes: Optional[Sequence[str]] = None,
    ) -> List[Detection]:
        """Run YOLO and return detections filtered to class names in `classes` if provided."""
        res = self.model.predict(
            source=frame_bgr,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=self._device,
            half=self._use_half,
            verbose=bool(self.verbose),
        )
        if not res:
            return []
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        boxes = r0.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clsi = boxes.cls.detach().cpu().numpy().astype(int)
        names = self.model.names  # id -> name
        H, W = frame_bgr.shape[:2]

        out: List[Detection] = []
        for (x1, y1, x2, y2), c, ci in zip(xyxy, confs, clsi):
            name = names.get(int(ci), str(int(ci)))
            if classes is not None and name not in classes:
                continue
            xy = clamp_xyxy((float(x1), float(y1), float(x2), float(y2)), W, H)
            out.append(Detection(xyxy=xy, conf=float(c), cls=name))
        return out

    def detect_players(self, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> List[Detection]:
        return self.detect(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, classes=[COCO_PERSON])

    def detect_ball(self, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> List[Detection]:
        return self.detect(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, classes=[COCO_SPORTS_BALL])
