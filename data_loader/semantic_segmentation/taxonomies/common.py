from dataclasses import dataclass
from enum import Enum
from typing import Generic, Iterable, TypeVar
import numpy as np

@dataclass(frozen=True)
class SourceClass:
    raw_id: int
    name: str
    
@dataclass(frozen=True)
class ProjectClass:
    key: str
    name: str
    description: str = ""

@dataclass(frozen=True)
class ColorClass:
    project_class: ProjectClass
    rgb: np.ndarray

IGNORE_LABEL = 255