from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .img import Splimage


def write_video(
    frames: Iterable[Splimage],
    path: str | Path,
    fps: float = 30,
) -> None:
    writer = None

    try:
        for i, frame in enumerate(frames):
            frame = np.asarray(frame.to_pil().convert("RGB"))

            if i == 0:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
