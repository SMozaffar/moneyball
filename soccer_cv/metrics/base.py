from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Dict, List, Optional
import numpy as np

from soccer_cv.types import FrameResult

class Metric(Protocol):
    def on_frame(self, fr: FrameResult) -> None: ...
    def finalize(self) -> Dict[str, Any]: ...

@dataclass
class MetricRegistry:
    metrics: List[Metric]

    def on_frame(self, fr: FrameResult) -> None:
        for m in self.metrics:
            m.on_frame(fr)

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for m in self.metrics:
            out[m.__class__.__name__] = m.finalize()
        return out
