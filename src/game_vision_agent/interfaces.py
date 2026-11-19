"""Typed, model-independent contracts for observation and simulator workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    def normalized(self, frame_width: int, frame_height: int) -> Box:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Box(self.x / frame_width, self.y / frame_height,
                   self.width / frame_width, self.height / frame_height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    identity: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    geometry: Box | None = None


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: Box
    normalized_bbox: Box
    center: tuple[float, float]
    frame_identity: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Observation:
    frame: Frame
    detections: tuple[Detection, ...] = ()
    change: object | None = None


@dataclass(frozen=True)
class GameState:
    screen: str | None = None
    values: dict[str, object] = field(default_factory=dict)
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    description: str
    priority: float = 0.0
    confidence: float = 0.0
    preconditions: tuple[str, ...] = ()
    expected_outcome: str | None = None
    timeout_seconds: float = 5.0
    source_frame_identity: str | None = None


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    executed: bool
    observed_outcome: str | None = None
    recovery_decision: str | None = None


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...
    def close(self) -> None: ...


class WindowLocator(Protocol):
    def list_windows(self) -> Sequence[WindowInfo]: ...
    def get_window(self, hwnd: int) -> WindowInfo | None: ...


class ChangeDetector(Protocol):
    def compare(self, before: np.ndarray | None, after: np.ndarray) -> ChangeResult: ...


class ObjectDetector(Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]: ...


class ActionExecutor(Protocol):
    def execute(self, proposal: ActionProposal, *, simulator: bool = False) -> ActionOutcome: ...


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    x: int
    y: int
    width: int
    height: int
    visible: bool
    minimized: bool
    client_x: int
    client_y: int
    client_width: int
    client_height: int


@dataclass(frozen=True)
class ChangeResult:
    magnitude: float
    changed: bool
    changed_region: Box | None
    changed_fraction: float


class Yolo26OnnxAdapter:
    """Extension point; refuses to invent detections until a real backend exists."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path

    def detect(self, frame: Frame) -> Sequence[Detection]:
        raise NotImplementedError("YOLO26n ONNX inference is planned, not implemented")
