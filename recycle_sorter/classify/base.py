from __future__ import annotations

from typing import Protocol

from ..types import Classification, Frame, ObjectObservation


class Classifier(Protocol):
    """The one piece that changes between stages (color -> shape -> material -> brand)."""

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification: ...
