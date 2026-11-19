from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from game_vision_agent.interfaces import WindowInfo


class WindowsWindowLocator:
    """Native user32 enumeration. Coordinates use physical pixels when per-monitor DPI is enabled."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows window discovery is only available on Windows")
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass

    def list_windows(self) -> list[WindowInfo]:
        result: list[WindowInfo] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd: int, _: int) -> bool:
            if not self.user32.IsWindowVisible(hwnd):
                return True
            length = self.user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, title, length + 1)
            rect = wintypes.RECT()
            if not self.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            client = wintypes.RECT()
            if not self.user32.GetClientRect(hwnd, ctypes.byref(client)):
                return True
            point = wintypes.POINT(0, 0)
            self.user32.ClientToScreen(hwnd, ctypes.byref(point))
            result.append(WindowInfo(int(hwnd), title.value, rect.left, rect.top,
                rect.right - rect.left, rect.bottom - rect.top,
                bool(self.user32.IsWindowVisible(hwnd)), bool(self.user32.IsIconic(hwnd)),
                point.x, point.y, client.right-client.left, client.bottom-client.top))
            return True

        self.user32.EnumWindows(callback_type(visit), 0)
        return result

    def get_window(self, hwnd: int) -> WindowInfo | None:
        return next((w for w in self.list_windows() if w.hwnd == hwnd), None)
