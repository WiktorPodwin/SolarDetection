from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class Mask:
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
    roof_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    ground_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    rectangle_shape: Optional[Tuple[int, int, int, int]] = None

@dataclass
class Depth:
    depth: np.ndarray = field(default_factory=lambda: np.array([]))
    focal_length: float = 0.0

@dataclass
class Image(Mask, Depth):
    name: str = ""
    location: str = ""
    size: int = 0
    metadata: dict = field(default_factory=dict)
    rgb: np.ndarray = field(default_factory=lambda: np.array([]))
    segmentation: np.ndarray = field(default_factory=lambda: np.array([]))
    classification: str = ""

    @property
    def masked_roof(self):
        # Apply the mask to the RGB image
        return np.concatenate((self.rgb, self.roof_mask), axis=-1)

    @property
    def masked_ground(self):
        # Apply the mask to the RGB image
        return np.concatenate((self.rgb, self.ground_mask), axis=-1)

    @property
    def describe(self):
        print(f"Image {self.name}")
        print(f"Location: {self.location}")
        print(f"Size: {self.size}")
        print(f"Metadata: {self.metadata}")
        print(f"RGB shape: {self.rgb.shape}")
        print(f"Depth shape: {self.depth.shape}")
        print(f"Focal length: {self.focal_length}")
        print(f"Segmentation shape: {self.segmentation.shape}")
        print(f"Classification: {self.classification}")
        print(f"Rectangle shape: {self.rectangle_shape}")
        return self
