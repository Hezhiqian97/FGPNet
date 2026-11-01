# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from FGP.engine.model import Model
from FGP.models import yolo
from FGP.nn.tasks import DetectionModel, SegmentationModel


class YOLO(Model):
    """YOLO (You Only Look Once) model.

    Unified entry point for the FGPNet instance segmentation models, covering object detection and instance
    segmentation tasks.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained detection model
        >>> model = YOLO("yolo26n.pt")

        Load a pretrained segmentation model
        >>> model = YOLO("yolo26n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment'. Defaults
                to auto-detection based on model.
            verbose (bool): Display model info on load.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
        }
