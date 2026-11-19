from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from game_vision_agent.benchmark.metrics import Metrics
from game_vision_agent.capture.mss_source import WindowCaptureSource
from game_vision_agent.capture.windows import WindowsWindowLocator
from game_vision_agent.perception.change import NumpyChangeDetector
from game_vision_agent.replay.sources import ReplayFrameSource, SyntheticFrameSource


def _save_image(image: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="game-vision", description="Read-only visual observation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("windows", help="List visible top-level windows").add_argument("action", choices=["list"])
    capture = sub.add_parser("capture", help="Capture visible frames from one selected window")
    capture.add_argument("--hwnd", type=lambda v: int(v, 0), required=True)
    capture.add_argument("--count", type=int, default=20)
    capture.add_argument("--interval", type=float, default=2.0)
    capture.add_argument("--out", type=Path, default=Path("local-data"))
    replay = sub.add_parser("replay", help="Read images from a local frame directory")
    replay.add_argument("--frames", type=Path, required=True)
    bench = sub.add_parser("benchmark", help="Run an offline synthetic benchmark")
    bench.add_argument("--source", choices=["synthetic"], default="synthetic")
    bench.add_argument("--duration", type=float, default=30.0)
    bench.add_argument("--out", type=Path, default=Path("benchmark-report.json"))
    args = parser.parse_args()

    if args.command == "windows":
        try:
            windows = WindowsWindowLocator().list_windows()
        except OSError as exc:
            parser.error(str(exc))
        if not windows:
            print("No visible top-level windows found.")
        for w in windows:
            print(f"{w.hwnd:#x}\t{w.client_width}x{w.client_height} at ({w.client_x},{w.client_y})\t"
                  f"visible={w.visible} minimized={w.minimized}\t{w.title!r}")
    elif args.command == "capture":
        if args.count < 1 or args.interval < 0:
            parser.error("--count must be positive and --interval non-negative")
        try:
            source = WindowCaptureSource(args.hwnd)
        except (OSError, RuntimeError) as exc:
            parser.error(str(exc))
        args.out.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(args.count):
                start = time.perf_counter()
                frame = source.read()
                if frame is None:
                    print(f"Selected window {args.hwnd:#x} is missing, minimized, or has invalid bounds.")
                    break
                path = args.out / f"frame-{i:06d}.png"
                _save_image(frame.image, path)
                print(f"Saved {path} ({frame.image.shape[1]}x{frame.image.shape[0]})")
                time.sleep(max(0, args.interval - (time.perf_counter() - start)))
        finally:
            source.close()
    elif args.command == "replay":
        if not args.frames.is_dir():
            parser.error(f"frame directory does not exist: {args.frames}")
        source = ReplayFrameSource(args.frames)
        detector = NumpyChangeDetector()
        previous = None
        count = 0
        while (frame := source.read()) is not None:
            change = detector.compare(previous, frame.image)
            print(f"{frame.identity}: change={change.magnitude:.3f} changed={change.changed}")
            previous = frame.image
            count += 1
        print(f"Replay complete: {count} frame(s).")
    else:
        if args.duration <= 0:
            parser.error("--duration must be positive")
        source = SyntheticFrameSource()
        detector = NumpyChangeDetector(threshold=0.25)
        metrics = Metrics()
        previous = None
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            frame = source.read()
            metrics.capture_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            result = detector.compare(previous, frame.image)
            metrics.detection_ms.append((time.perf_counter() - t1) * 1000)
            if not result.changed:
                metrics.skipped += 1
            previous = frame.image
            metrics.frames += 1
            time.sleep(1 / source.fps)
        report = metrics.report()
        report.update({"source": "synthetic", "steady_state_note": "includes loop processing over requested duration"})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
