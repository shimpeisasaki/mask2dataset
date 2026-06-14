from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


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

    @staticmethod
    def _v360_expr(*, out_size: int, fov: float, yaw: float, pitch: float, interp: str) -> str:
        return ":".join(
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
                f"interp={interp}",
            ]
        )

    def project_many_rgb_from_array(
        self,
        input_rgb: np.ndarray,
        specs: Sequence[ViewSpec],
        *,
        out_size: int,
        fov: float,
    ) -> List[np.ndarray]:
        """Project multiple rectilinear views from in-memory equirect RGB.

        Uses a single ffmpeg process with rawvideo pipe input/output to avoid
        per-view PNG I/O and repeated process overhead.
        """
        if input_rgb.ndim != 3 or input_rgb.shape[2] != 3:
            raise ValueError("input_rgb must be HxWx3")
        if input_rgb.dtype != np.uint8:
            raise ValueError("input_rgb must be uint8")
        if not specs:
            return []

        h, w = int(input_rgb.shape[0]), int(input_rgb.shape[1])
        n = len(specs)

        parts: List[str] = []
        if n == 1:
            spec = specs[0]
            v360 = self._v360_expr(
                out_size=out_size,
                fov=fov,
                yaw=spec.yaw,
                pitch=spec.pitch,
                interp="line",
            )
            parts.append(f"[0:v]format=rgb24,v360={v360}[stack]")
        else:
            split_out = "".join([f"[s{i}]" for i in range(n)])
            parts.append(f"[0:v]format=rgb24,split={n}{split_out}")
            for i, spec in enumerate(specs):
                v360 = self._v360_expr(
                    out_size=out_size,
                    fov=fov,
                    yaw=spec.yaw,
                    pitch=spec.pitch,
                    interp="line",
                )
                parts.append(f"[s{i}]v360={v360}[o{i}]")

            layout = "|".join([f"{i * int(out_size)}_0" for i in range(n)])
            inputs = "".join([f"[o{i}]" for i in range(n)])
            parts.append(f"{inputs}xstack=inputs={n}:layout={layout}[stack]")

        filter_complex = ";".join(parts)

        cmd: List[str] = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{w}x{h}",
            "-r",
            "1",
            "-i",
            "pipe:0",
            "-filter_complex",
            filter_complex,
            "-map",
            "[stack]",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

        proc = subprocess.run(cmd, input=input_rgb.tobytes(), capture_output=True)
        if proc.returncode != 0:
            err = (proc.stderr.decode("utf-8", errors="ignore") or "").strip()
            raise RuntimeError(f"ffmpeg v360 in-memory batch failed: {err}")

        out_w = int(out_size) * n
        expected = int(out_size) * out_w * 3
        raw = proc.stdout or b""
        if len(raw) < expected:
            raise RuntimeError(
                f"ffmpeg v360 in-memory batch returned short frame: got={len(raw)} expected={expected}"
            )

        sheet = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((int(out_size), out_w, 3)).copy()
        tiles: List[np.ndarray] = []
        for i in range(n):
            x0 = i * int(out_size)
            x1 = (i + 1) * int(out_size)
            tiles.append(sheet[:, x0:x1, :].copy())
        return tiles

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
            v360 = self._v360_expr(
                out_size=out_size,
                fov=fov,
                yaw=spec.yaw,
                pitch=spec.pitch,
                interp="line",
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
            v360 = self._v360_expr(
                out_size=out_size,
                fov=fov,
                yaw=spec.yaw,
                pitch=spec.pitch,
                interp="near",
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
        v360 = self._v360_expr(
            out_size=out_size,
            fov=fov,
            yaw=yaw,
            pitch=pitch,
            interp="line",
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
        v360 = self._v360_expr(
            out_size=out_size,
            fov=fov,
            yaw=yaw,
            pitch=pitch,
            interp="near",
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
