"""
Shared RF-DETR inference core for the Waste Detection app.
Both models use RF-DETR-Medium at resolution 704 — same architecture family.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv
from PIL import Image


# --------------------------------------------------------------------------- #
# Class taxonomies
# --------------------------------------------------------------------------- #

TACO_CLASSES: Dict[int, str] = {
    0: "taco-waste",
    1: "Bottle",
    2: "Bottle cap",
    3: "Can",
    4: "Cigarette",
    5: "Cup",
    6: "Lid",
    7: "Other",
    8: "Plastic bag and wrapper",
    9: "Pop tab",
    10: "Straw",
}

MULTIWASTE_CLASSES: Dict[int, str] = {
    1: "cardboard",
    2: "carton packaging",
    3: "cigarette",
    4: "clean paper",
    5: "clear plastic",
    6: "contaminated paper",
    7: "food packaging",
    8: "food scraps",
    9: "glass",
    10: "medical waste",
    11: "metal",
    12: "paper bag",
    13: "paper cup",
    14: "plastic bottle",
    15: "plastic container",
    16: "plastic cup",
    17: "plastic lid",
    18: "plastic packaging",
    19: "plastic utensil",
    20: "printed cardboard",
    21: "sanitary waste",
    22: "straw",
    23: "styrofoam",
    24: "wood",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint: Path
    classes: Dict[int, str] = field(repr=False)
    dataset: str = ""
    val_map: float = 0.0
    resolution: int = 704

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def build_registry(models_dir: Path) -> Dict[str, ModelSpec]:
    return {
        "taco": ModelSpec(
            key="taco",
            display_name="TACO RF-DETR",
            checkpoint=models_dir / "taco_rfdetr.pth",
            classes=TACO_CLASSES,
            dataset="TACO v3 - 10 classes (original taxonomy)",
            val_map=0.419,
            resolution=704,
        ),
        "multiwaste": ModelSpec(
            key="multiwaste",
            display_name="MultipleWaste RF-DETR",
            checkpoint=models_dir / "multiwaste_rfdetr.pth",
            classes=MULTIWASTE_CLASSES,
            dataset="Multi-Class Waste v3 - 24 material classes",
            val_map=0.717,
            resolution=704,
        ),
    }


@dataclass
class DetectionResult:
    annotated: np.ndarray
    labels: List[str]
    class_counts: Dict[str, int]
    inference_ms: float
    mean_confidence: float

    @property
    def count(self) -> int:
        return len(self.labels)


class WasteDetector:
    """Both TACO v3 and MultipleWaste v3 use RF-DETR-Medium — single code path."""

    def __init__(self, spec: ModelSpec) -> None:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {spec.checkpoint}\n"
                f"Place '{spec.checkpoint.name}' inside the models/ folder."
            )

        from rfdetr import RFDETRMedium

        self.spec = spec
        self._model = RFDETRMedium(
            pretrain_weights=str(spec.checkpoint),
            resolution=spec.resolution,
        )
        self._model.optimize_for_inference()

        self._box = sv.BoxAnnotator(thickness=3)
        self._label = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)

    def predict(self, image: Image.Image, threshold: float = 0.5) -> DetectionResult:
        rgb = image.convert("RGB")
        scene = np.array(rgb)

        start = time.perf_counter()
        detections = self._model.predict(rgb, threshold=threshold)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if len(detections.class_id) > 0:
            detections = detections.with_nms(threshold=0.5, class_agnostic=False)

        labels = [
            f"{self._name(class_id)} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        counts: Dict[str, int] = {}
        for class_id in detections.class_id:
            name = self._name(class_id)
            counts[name] = counts.get(name, 0) + 1

        annotated = self._box.annotate(scene.copy(), detections)
        annotated = self._label.annotate(annotated, detections, labels=labels)

        confidences = np.asarray(detections.confidence, dtype=float)
        mean_conf = float(confidences.mean()) if confidences.size else 0.0

        return DetectionResult(
            annotated=annotated,
            labels=labels,
            class_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            inference_ms=elapsed_ms,
            mean_confidence=mean_conf,
        )

    def _name(self, class_id: int) -> str:
        return self.spec.classes.get(int(class_id), f"id:{int(class_id)}")


def load_detectors(specs: Tuple[ModelSpec, ...]) -> Dict[str, WasteDetector]:
    return {spec.key: WasteDetector(spec) for spec in specs}
