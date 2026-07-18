from dataclasses import dataclass
from enum import Enum
from typing import Generic, Iterable, TypeVar
import numpy as np

import os

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data_loader.semantic_segmentation.taxonomies.common import SourceClass, ProjectClass, IGNORE_LABEL
from data_loader.semantic_segmentation.taxonomies.accessibility_class import AccessibilityClass

S = TypeVar("S", bound=Enum)
T = TypeVar("T", bound=Enum)

@dataclass
class MappingSpec(Generic[S, T]):
    name: str
    source_enum: type[S]
    target_enum: type[T]
    target_classes: tuple[T, ...]
    assignments: dict[S, T]
    unrepresented_targets: dict[T, str]
    ignore_label: int = IGNORE_LABEL
    
    def __post_init__(self):
        self.validate()
    
    @property
    def n_classes(self) -> int:
        return len(self.target_classes)
    
    @property
    def target_to_train_id(self) -> dict[AccessibilityClass, int]:
        """
        Returns a dictionary mapping each target class to its corresponding train ID.
        """
        return {cls: idx for idx, cls in enumerate(self.target_classes)}
    
    @property
    def class_names(self) -> list[str]:
        return [cls.value.name for cls in self.target_classes]
    
    def validate(self):
        valid_targets = set(self.target_classes)
        for source_cls, target_cls in self.assignments.items():
            if not isinstance(source_cls, self.source_enum):
                raise ValueError(f"Source class {source_cls} is not a valid member of {self.source_enum.__name__}")
            if not isinstance(target_cls, self.target_enum):
                raise ValueError(f"Target class {target_cls} is not a valid member of {self.target_enum.__name__}")
            if target_cls not in valid_targets:
                raise ValueError(f"Target class {target_cls} is not a valid member of target_classes")
    
    def compile_label_map(self, num_source_labels: int | None = None) -> np.ndarray:
        # self.validate()
        
        if num_source_labels is None:
            # max_raw_id = max(source.value.raw_id for source in self.source_enum)
            # num_source_labels = max_raw_id + 1
            num_source_labels = 256  # Default to 256 if not provided
        
        lb_map = np.full(num_source_labels, self.ignore_label, dtype=np.uint8)
        target_to_id = self.target_to_train_id
        for source_class, target_class in self.assignments.items():
            raw_id = source_class.value.raw_id
            if raw_id >= num_source_labels:
                raise ValueError(f"Source class {source_class} has raw_id {raw_id} which exceeds the number of source labels {num_source_labels}")
            train_id = target_to_id[target_class]
            lb_map[raw_id] = train_id
        return lb_map
    
    def get_report(self) -> dict:
        return {
            "mapping_name": self.name,
            "source_dataset": self.source_dataset,
            "ignore_label": self.ignore_label,
            "n_classes": self.n_classes,
            "target_classes": [
                {
                    "train_id": idx,
                    "key": cls.key,
                    "name": cls.name,
                    "description": cls.description,
                }
                for idx, cls in self.train_id_to_target.items()
            ],
            "unrepresented_targets": [
                {
                    "key": cls.key,
                    "name": cls.name,
                    "description": cls.description,
                    "reason": reason,
                }
                for cls, reason in self.unrepresented_targets.items()
            ],
            "source_to_target": self.source_to_target_report(),
        }
        
class MappingBuilder:
    def __init__(
        self,
        name: str,
        source_enum: type[S],
        target_enum: type[T],
        target_classes: Iterable[T],
        ignore_label: int = IGNORE_LABEL,
    ):
        self.name = name
        self.source_enum = source_enum
        self.target_enum = target_enum
        self.target_classes = tuple(target_classes)
        self.ignore_label = ignore_label
        self.assignments: dict[S, T] = {}
        self.unrepresented_targets: dict[T, str] = {}
        
    def map(self, source: S, target: T) -> "MappingBuilder[S, T]":
        if not isinstance(source, self.source_enum):
            raise ValueError(f"Source class {source} is not a valid member of {self.source_enum.__name__}")
        if not isinstance(target, self.target_enum):
            raise ValueError(f"Target class {target} is not a valid member of {self.target_enum.__name__}")
        if target not in self.target_classes:
            raise ValueError(f"Target class {target} is not a valid member of target_classes")
        self.assignments[source] = target
        return self
    
    def map_many(self, sources: Iterable[S], target: T) -> "MappingBuilder[S, T]":
        for source in sources:
            self.map(source, target)
        return self
    
    def mark_unrepresented(self, target: T, reason: str = "") -> "MappingBuilder[S, T]":
        if target not in self.target_classes:
            raise ValueError(
                f"{target} cannot be marked unrepresented because it is not "
                f"in target_classes."
            )

        self.unrepresented_targets[target] = reason
        return self
    
    def build(self) -> MappingSpec[S, T]:
        return MappingSpec(
            name=self.name,
            source_enum=self.source_enum,
            target_enum=self.target_enum,
            target_classes=self.target_classes,
            assignments=self.assignments,
            unrepresented_targets=self.unrepresented_targets,
            ignore_label=self.ignore_label,
        )