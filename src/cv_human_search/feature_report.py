"""Structured reporting helpers for image analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageAnalysisReport:
    """Aggregated report for the still-image pipeline."""

    metadata: Dict[str, Any]
    histogram_peak: Optional[int] = None
    face_boxes: List[tuple[int, int, int, int]] = field(default_factory=list)
    feature_summaries: Dict[str, Any] = field(default_factory=dict)

