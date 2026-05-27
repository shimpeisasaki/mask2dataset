from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


@dataclass(frozen=True)
class ClassMap:
    """Maps ADE20K label names/ids into compact dataset ids (0..7) + ignore (255).

    Output labels are uint8 values:
      - 0..7 : trainable classes
      - 255  : ignore
    """

    id_to_name: Dict[int, str]
    ade_name_to_id: Dict[str, int]
    ade_id_to_dataset_id: Dict[int, int]
    ignore_id: int = 255

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return name.strip().lower()

    @classmethod
    def from_yaml(cls, path: Path, ade_id2label: Mapping[int, str]) -> "ClassMap":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("class map yaml must be a mapping")

        ignore_id = int(raw.get("ignore_id", 255))
        if ignore_id != 255:
            raise ValueError("ignore_id must be 255 for mmseg-style ignore")

        classes = raw.get("classes")
        if not isinstance(classes, dict):
            raise ValueError("yaml must contain 'classes' mapping: {0: {name, ade20k:[...]}, ...}")

        ade_name_to_id: Dict[str, int] = {
            cls._normalize_label_name(v): int(k) for k, v in ade_id2label.items()
        }

        id_to_name: Dict[int, str] = {}
        ade_id_to_dataset_id: Dict[int, int] = {}

        for dataset_id_raw, spec in classes.items():
            dataset_id = int(dataset_id_raw)
            if dataset_id < 0 or dataset_id > 7:
                raise ValueError("dataset class ids must be in 0..7")
            if not isinstance(spec, dict):
                raise ValueError(f"classes[{dataset_id}] must be a mapping")

            name = str(spec.get("name", f"class{dataset_id}"))
            id_to_name[dataset_id] = name

            ade_list = spec.get("ade20k", [])
            if not isinstance(ade_list, list):
                raise ValueError(f"classes[{dataset_id}].ade20k must be a list")

            for ade_name in ade_list:
                key = cls._normalize_label_name(str(ade_name))
                ade_id = ade_name_to_id.get(key)
                if ade_id is None:
                    raise ValueError(f"unknown ADE20K label name in yaml: '{ade_name}'")
                ade_id_to_dataset_id[ade_id] = dataset_id

        unmapped = int(raw.get("unmapped", 255))
        if unmapped not in (*range(0, 8), 255):
            raise ValueError("unmapped must be 0..7 (a dataset class id) or 255 (ignore)")

        # Store unmapped policy as a pseudo entry using key -1.
        ade_id_to_dataset_id[-1] = unmapped

        return cls(
            id_to_name=id_to_name,
            ade_name_to_id=ade_name_to_id,
            ade_id_to_dataset_id=ade_id_to_dataset_id,
            ignore_id=ignore_id,
        )

    def dataset_id_for_ade_id(self, ade_id: int) -> int:
        if ade_id in self.ade_id_to_dataset_id:
            return int(self.ade_id_to_dataset_id[ade_id])
        return int(self.ade_id_to_dataset_id.get(-1, self.ignore_id))

    def summarize(self) -> str:
        lines: List[str] = []
        for dataset_id in sorted(self.id_to_name.keys()):
            lines.append(f"{dataset_id}: {self.id_to_name[dataset_id]}")
        lines.append(f"unmapped -> {self.ade_id_to_dataset_id.get(-1, self.ignore_id)}")
        return " | ".join(lines)
