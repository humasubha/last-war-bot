from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from game_vision_agent.capture.windows import WindowsWindowLocator
from game_vision_agent.interfaces import Box, Frame


class WindowCaptureSource:
    """Read-only client-area screen crop; minimized or obscured content is not recoverable."""

    def __init__(self, hwnd: int):
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("MSS is required; install game-vision-agent") from exc
        self.hwnd = hwnd
        self.locator = WindowsWindowLocator()
        self._sct = mss.MSS()
        self._sequence = 0

    def read(self) -> Frame | None:
        info = self.locator.get_window(self.hwnd)
        if info is None or info.minimized or info.client_width <= 0 or info.client_height <= 0:
            return None
        # Re-query each frame so window movement and resizing are reflected.
        shot = self._sct.grab({"left": info.client_x, "top": info.client_y,
                              "width": info.client_width, "height": info.client_height})
        bgra = np.asarray(shot)
        rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
        self._sequence += 1
        return Frame(rgb, f"hwnd:{self.hwnd}:{self._sequence}",
                     datetime.now(UTC), Box(info.client_x, info.client_y,
                                                     info.client_width, info.client_height))

    def close(self) -> None:
        self._sct.close()
