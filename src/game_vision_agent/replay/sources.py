from __future__ import annotations

from pathlib import Path

import numpy as np

from game_vision_agent.interfaces import Frame


class SyntheticFrameSource:
    def __init__(self, width: int = 320, height: int = 180, fps: float = 5.0):
        self.width, self.height, self.fps = width, height, fps
        self.index = 0

    def read(self) -> Frame:
        img = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        x = (self.index * 13) % max(1, self.width - 32)
        img[40:72, x:x+32] = (40, 180, 240)
        frame = Frame(img, f"synthetic:{self.index}")
        self.index += 1
        return frame

    def close(self) -> None:
        pass


class ReplayFrameSource:
    def __init__(self, directory: Path):
        self.paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        self.index = 0

    def read(self) -> Frame | None:
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        try:
            from PIL import Image
            image = np.asarray(Image.open(path).convert("RGB"))
        except ImportError as exc:
            raise RuntimeError("Pillow is needed to replay image files; install the dev extra") from exc
        self.index += 1
        return Frame(image, f"replay:{path.name}")

    def close(self) -> None:
        pass
