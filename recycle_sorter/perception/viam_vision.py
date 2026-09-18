from __future__ import annotations

import logging
from typing import Any

import numpy as np
from viam.services.vision import VisionClient

from ..types import Frame, ObjectObservation
from .frames import to_world
from .pcd import parse_pcd
from .segment import finish, observe

log = logging.getLogger(__name__)

DUPLICATE_MM = 25.0  # two services reporting centroids this close have found the same item


async def detect(
    services: dict[str, VisionClient],
    camera_name: str,
    frame: Frame,
    workspace: dict[str, Any],
    use_detection_label: bool = False,
    timeout: float | None = None,
) -> list[ObjectObservation]:
    """Find objects with Viam vision services (get_object_point_clouds).

    services: class label -> vision service that finds objects of that class. A color
    detector + segmenter only reports its one target color, so sorting several classes
    means one service per class, and the class is simply which service saw the item.
    With use_detection_label the label comes from the detection itself instead (for an
    ML detector that names what it found).

    Only the raw points are taken from the service. Size, top height and grasp angle are
    computed from them by the shared observe(), not from the bounding box center.
    """
    found: list[ObjectObservation] = []
    for label, service in services.items():
        try:
            pcos = await service.get_object_point_clouds(camera_name, timeout=timeout)
        except Exception as e:
            log.warning("vision service for %r failed: %s", label, e)
            continue
        for pco in pcos:
            points = parse_pcd(pco.point_cloud)
            if not len(points):
                continue
            # Segmenters answer in the camera's frame; anything else is already world.
            if pco.geometries.reference_frame in ("", camera_name):
                points = to_world(points, frame.cam_to_world)
            geoms = pco.geometries.geometries
            name = geoms[0].label if (use_detection_label and geoms and geoms[0].label) else label
            o = observe(points, frame, workspace, source_label=name)
            if o is not None:
                found.append(o)

    # Overlapping detectors (red vs orange) can both claim one item: keep the fuller cloud.
    found.sort(key=lambda o: -len(o.points))
    unique: list[ObjectObservation] = []
    for o in found:
        if all(np.linalg.norm(o.centroid - u.centroid) > DUPLICATE_MM for u in unique):
            unique.append(o)
    return finish(unique, workspace)
