from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class Image:
    name: str
    location: str = ""
    size: int = 0
    metadata: dict = field(default_factory=dict)
    rgb: np.ndarray = field(default_factory=lambda: np.array([]))
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    focal_length: float = .0
    segmentation: np.ndarray = field(default_factory=lambda: np.array([]))
    classification: str = ""
    rectangle_shape: Optional[Tuple[int, int, int, int]] = None
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
