"""Audio slicing for the golden labeler (ffmpeg wrapper).

Long meetings (AMI is ~20-40 min) must NOT be re-sent whole on every transcript
window — that reprocesses the entire recording per call. Instead we cut each window
to its own time span and send only that clip. ffmpeg is a light local dependency
(no model), so this is safe to run anywhere and is unit-testable offline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def slice_audio(
    in_path: str | Path,
    start: float,
    end: float,
    out_path: str | Path,
    pad: float = 0.5,
    sample_rate: int = 16000,
) -> Path:
    """Cut [start-pad, end+pad] from `in_path` into a 16 kHz mono wav.

    Downmix + downsample keeps the upload small. Returns the output path.
    """
    start_eff = max(0.0, start - pad)
    dur = max(0.05, (end - start) + 2 * pad)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-ss", f"{start_eff:.3f}",     # seek before -i (fast)
        "-i", str(in_path),
        "-t", f"{dur:.3f}",            # duration, relative to the seek point
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-loglevel", "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def window_time_span(turns) -> tuple[float, float]:
    """[min start, max end] over a list of Turn objects."""
    starts = [t.start for t in turns]
    ends = [t.end for t in turns]
    return (min(starts), max(ends)) if turns else (0.0, 0.0)
