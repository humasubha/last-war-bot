from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from game_vision_agent.interfaces import Box, ChangeResult


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int


class NumpyChangeDetector:
    """Downsampled mean absolute difference with optional ignore regions."""

    def __init__(self, threshold: float = 3.0, sample_step: int = 4,
                 ignore_regions: tuple[Region, ...] = (), roi: Region | None = None):
        self.threshold = threshold
        self.sample_step = max(1, sample_step)
        self.ignore_regions = ignore_regions
        self.roi = roi

    def compare(self, before: np.ndarray | None, after: np.ndarray) -> ChangeResult:
        if before is None:
            return ChangeResult(0.0, False, None, 0.0)
        if before.shape != after.shape:
            return ChangeResult(255.0, True, Box(0, 0, after.shape[1], after.shape[0]), 1.0)
        a = before[::self.sample_step, ::self.sample_step, ...].astype(np.int16)
        b = after[::self.sample_step, ::self.sample_step, ...].astype(np.int16)
        if a.ndim == 3:
            diff = np.abs(a - b).mean(axis=2)
        else:
            diff = np.abs(a - b)
        h, w = diff.shape
        ys = np.arange(h) * self.sample_step
        xs = np.arange(w) * self.sample_step
        valid = np.ones((h, w), dtype=bool)
        if self.roi:
            valid &= ((xs[None, :] >= self.roi.x) & (xs[None, :] < self.roi.x + self.roi.width)
                      & (ys[:, None] >= self.roi.y) & (ys[:, None] < self.roi.y + self.roi.height))
        for r in self.ignore_regions:
            valid &= ~((xs[None, :] >= r.x) & (xs[None, :] < r.x + r.width)
                       & (ys[:, None] >= r.y) & (ys[:, None] < r.y + r.height))
        vals = diff[valid]
        magnitude = float(vals.mean()) if vals.size else 0.0
        changed_mask = (diff > self.threshold) & valid
        count = int(changed_mask.sum())
        fraction = count / max(1, int(valid.sum()))
        region = None
        if count:
            cy, cx = np.where(changed_mask)
            x0, y0 = int(cx.min() * self.sample_step), int(cy.min() * self.sample_step)
            x1, y1 = int((cx.max() + 1) * self.sample_step), int((cy.max() + 1) * self.sample_step)
            region = Box(x0, y0, min(x1, after.shape[1]) - x0, min(y1, after.shape[0]) - y0)
        return ChangeResult(magnitude, magnitude >= self.threshold and count > 0, region, fraction)
