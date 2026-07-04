from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Mapping

import yaml


@dataclass(frozen=True)
class ClassMap:
    """Maps ADE20K labels into dataset ids (0..254) + ignore (255)."""

    id_to_name: Dict[int, str]
    ade_name_to_id: Dict[str, int]
    ade_id_to_dataset_id: Dict[int, int]
    ignore_id: int = 255

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _compact_label_name(name: str) -> str:
        # Compact form improves matching robustness for names like "street light" vs "streetlight".
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

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

        ade_name_to_id: Dict[str, int] = {}
        for ade_id, raw_name in ade_id2label.items():
            did = int(ade_id)
            name = str(raw_name)
            norm = cls._normalize_label_name(name)
            compact = cls._compact_label_name(name)
            if norm:
                ade_name_to_id.setdefault(norm, did)
            if compact:
                ade_name_to_id.setdefault(compact, did)

            # Some ADE labels can appear as comma-separated aliases in model configs.
            for token in name.split(","):
                t_norm = cls._normalize_label_name(token)
                t_compact = cls._compact_label_name(token)
                if t_norm:
                    ade_name_to_id.setdefault(t_norm, did)
                if t_compact:
                    ade_name_to_id.setdefault(t_compact, did)

        id_to_name: Dict[int, str] = {}
        ade_id_to_dataset_id: Dict[int, int] = {}

        for dataset_id_raw, spec in classes.items():
            dataset_id = int(dataset_id_raw)
            if dataset_id < 0 or dataset_id > 254:
                raise ValueError("dataset class ids must be in 0..254")
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
                    ade_id = ade_name_to_id.get(cls._compact_label_name(key))
                if ade_id is None:
                    # Offer nearby candidates for easier map maintenance.
                    compact = cls._compact_label_name(key)
                    candidates = sorted({k for k in ade_name_to_id.keys() if compact and compact in k})[:8]
                    hint = f" candidates={candidates}" if candidates else ""
                    raise ValueError(f"unknown ADE20K label name in yaml: '{ade_name}'.{hint}")
                ade_id_to_dataset_id[ade_id] = dataset_id

        unmapped = int(raw.get("unmapped", 255))
        if not ((0 <= unmapped <= 254) or unmapped == 255):
            raise ValueError("unmapped must be 0..254 (a dataset class id) or 255 (ignore)")

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
