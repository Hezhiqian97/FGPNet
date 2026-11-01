# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from FGP.models.yolo import detect, segment

from .model import YOLO

__all__ = "YOLO", "detect", "segment"
