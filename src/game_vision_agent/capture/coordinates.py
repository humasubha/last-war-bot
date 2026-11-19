"""Conversions among frame pixels, client pixels, and desktop physical pixels."""

from game_vision_agent.interfaces import Box, WindowInfo


def frame_to_desktop(box: Box, window: WindowInfo, frame_width: int | None = None,
                     frame_height: int | None = None) -> Box:
    """Map a box in a client-sized frame to physical desktop coordinates."""
    sx = window.client_width / max(1, frame_width or window.client_width)
    sy = window.client_height / max(1, frame_height or window.client_height)
    return Box(window.client_x + box.x * sx, window.client_y + box.y * sy,
               box.width * sx, box.height * sy)


def desktop_to_client(box: Box, window: WindowInfo, frame_width: int | None = None,
                      frame_height: int | None = None) -> Box:
    """Map desktop physical coordinates into selected window client pixels."""
    sx = (frame_width or window.client_width) / max(1, window.client_width)
    sy = (frame_height or window.client_height) / max(1, window.client_height)
    return Box((box.x - window.client_x) * sx, (box.y - window.client_y) * sy,
               box.width * sx, box.height * sy)
