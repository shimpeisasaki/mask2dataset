from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


def _to_ffmpeg_angle(angle_deg: float) -> float:
    # ffmpeg v360 expects [-180, 180]
    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return a


@dataclass(frozen=True)
class ViewSpec:
    name: str
    yaw: float
    pitch: float


@dataclass(frozen=True)
class V360Projector:
    ffmpeg: str = "ffmpeg"

    def project_many_rgb(
        self,
        input_path: Path,
        specs: Sequence[ViewSpec],
        output_paths: Sequence[Path],
        *,
        out_size: int,
        fov: float,
    ) -> None:
        if len(specs) != len(output_paths):
            raise ValueError("specs and output_paths length mismatch")
        if not specs:
            return

        n = len(specs)
        split_out = "".join([f"[s{i}]" for i in range(n)])
        parts: List[str] = [f"[0:v]split={n}{split_out}"]
        for i, spec in enumerate(specs):
            v360 = ":".join(
                [
                    "input=e",
                    "output=rectilinear",
                    f"h_fov={float(fov)}",
                    f"v_fov={float(fov)}",
                    f"w={int(out_size)}",
                    f"h={int(out_size)}",
                    f"yaw={_to_ffmpeg_angle(spec.yaw)}",
                    f"pitch={_to_ffmpeg_angle(spec.pitch)}",
                    "roll=0",
                    "interp=line",
                ]
            )
            parts.append(f"[s{i}]v360={v360}[o{i}]")

        filter_complex = ";".join(parts)

        cmd: List[str] = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
        ]
        for i, outp in enumerate(output_paths):
            cmd.extend(["-map", f"[o{i}]", str(outp)])
        cmd.extend(["-frames:v", "1"])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg v360 batch failed: {err}")

        for outp in output_paths:
            if not outp.exists():
                raise RuntimeError(f"ffmpeg v360 batch failed to create: {outp}")

    def project_many_label_nearest(
        self,
        input_path: Path,
        specs: Sequence[ViewSpec],
        output_paths: Sequence[Path],
        *,
        out_size: int,
        fov: float,
    ) -> None:
        if len(specs) != len(output_paths):
            raise ValueError("specs and output_paths length mismatch")
        if not specs:
            return

        n = len(specs)
        split_out = "".join([f"[s{i}]" for i in range(n)])
        parts: List[str] = [f"[0:v]split={n}{split_out}"]
        for i, spec in enumerate(specs):
            v360 = ":".join(
                [
                    "input=e",
                    "output=rectilinear",
                    f"h_fov={float(fov)}",
                    f"v_fov={float(fov)}",
                    f"w={int(out_size)}",
                    f"h={int(out_size)}",
                    f"yaw={_to_ffmpeg_angle(spec.yaw)}",
                    f"pitch={_to_ffmpeg_angle(spec.pitch)}",
                    "roll=0",
                    "interp=near",
                ]
            )
            parts.append(f"[s{i}]v360={v360},format=gray[o{i}]")

        filter_complex = ";".join(parts)

        cmd: List[str] = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
        ]
        for i, outp in enumerate(output_paths):
            cmd.extend(["-map", f"[o{i}]", "-pix_fmt", "gray", str(outp)])
        cmd.extend(["-frames:v", "1"])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg v360(label) batch failed: {err}")

        for outp in output_paths:
            if not outp.exists():
                raise RuntimeError(f"ffmpeg v360(label) batch failed to create: {outp}")

    def project_rgb(
        self,
        input_path: Path,
        output_path: Path,
        out_size: int,
        fov: float,
        yaw: float,
        pitch: float,
    ) -> None:
        v360 = ":".join(
            [
                "input=e",
                "output=rectilinear",
                f"h_fov={float(fov)}",
                f"v_fov={float(fov)}",
                f"w={int(out_size)}",
                f"h={int(out_size)}",
                f"yaw={_to_ffmpeg_angle(yaw)}",
                f"pitch={_to_ffmpeg_angle(pitch)}",
                "roll=0",
                "interp=line",
            ]
        )

        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"v360={v360}",
            "-frames:v",
            "1",
            str(output_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not output_path.exists():
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg v360 failed: {err}")

    def project_label_nearest(
        self,
        input_path: Path,
        output_path: Path,
        out_size: int,
        fov: float,
        yaw: float,
        pitch: float,
    ) -> None:
        v360 = ":".join(
            [
                "input=e",
                "output=rectilinear",
                f"h_fov={float(fov)}",
                f"v_fov={float(fov)}",
                f"w={int(out_size)}",
                f"h={int(out_size)}",
                f"yaw={_to_ffmpeg_angle(yaw)}",
                f"pitch={_to_ffmpeg_angle(pitch)}",
                "roll=0",
                "interp=near",
            ]
        )

        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"v360={v360},format=gray",
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            str(output_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not output_path.exists():
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg v360(label) failed: {err}")
