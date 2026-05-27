from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class ClassMap:
    """Maps free-form text prompts into compact dataset ids (0..7) + ignore (255).

    This branch (feat/sam) uses FastSAM to propose masks, then assigns each mask to a
    dataset class by comparing the mask crop to `prompts` using CLIP.

    Output labels are uint8 values:
      - 0..7 : trainable classes
      - 255  : ignore
    """

    id_to_name: Dict[int, str]
    id_to_prompts: Dict[int, List[str]]
    unmapped: int
    ignore_id: int = 255

    @classmethod
    def from_yaml(cls, path: Path) -> "ClassMap":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("class map yaml must be a mapping")

        ignore_id = int(raw.get("ignore_id", 255))
        if ignore_id != 255:
            raise ValueError("ignore_id must be 255 for mmseg-style ignore")

        classes = raw.get("classes")
        if not isinstance(classes, dict):
            raise ValueError("yaml must contain 'classes' mapping: {0: {name, prompts:[...]}, ...}")

        id_to_name: Dict[int, str] = {}
        id_to_prompts: Dict[int, List[str]] = {}

        for dataset_id_raw, spec in classes.items():
            dataset_id = int(dataset_id_raw)
            if dataset_id < 0 or dataset_id > 7:
                raise ValueError("dataset class ids must be in 0..7")
            if not isinstance(spec, dict):
                raise ValueError(f"classes[{dataset_id}] must be a mapping")

            name = str(spec.get("name", f"class{dataset_id}"))
            id_to_name[dataset_id] = name

            prompts = spec.get("prompts")
            if prompts is None:
                # Backward-compat: allow old key name in this branch.
                prompts = spec.get("ade20k", [])
            if not isinstance(prompts, list):
                raise ValueError(f"classes[{dataset_id}].prompts must be a list")

            out_prompts: List[str] = []
            for p in prompts:
                s = str(p).strip()
                if s:
                    out_prompts.append(s)
            if not out_prompts:
                out_prompts = [name]
            id_to_prompts[dataset_id] = out_prompts

        unmapped = int(raw.get("unmapped", 255))
        if unmapped not in (*range(0, 8), 255):
            raise ValueError("unmapped must be 0..7 (a dataset class id) or 255 (ignore)")

        return cls(
            id_to_name=id_to_name,
            id_to_prompts=id_to_prompts,
            unmapped=unmapped,
            ignore_id=ignore_id,
        )

    def summarize(self) -> str:
        lines: List[str] = []
        for dataset_id in sorted(self.id_to_name.keys()):
            lines.append(f"{dataset_id}: {self.id_to_name[dataset_id]}")
        lines.append(f"unmapped -> {self.unmapped}")
        return " | ".join(lines)
