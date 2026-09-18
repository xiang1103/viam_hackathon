from __future__ import annotations

from ..types import Classification, Frame, ObjectObservation
from .base import Classifier


class VisionLabelClassifier:
    """The class is whatever the Viam vision service that found the object says it is.

    Objects that did not come from a vision service (the OpenCV fallback, or frames
    recorded without one) are handed to `fallback` instead.
    """

    def __init__(self, fallback: Classifier):
        self.fallback = fallback

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        if obs.source_label:
            return Classification(obs.source_label, 1.0, {"source": "viam-vision"})
        return await self.fallback.classify(obs, frame)
