# game-vision-agent

An offline-first, read-only Windows screen observation foundation. Phase 1 provides window discovery, visible client-area capture, low-cost change detection, synthetic/replay sources, metrics, and safe model/action contracts. It does not automate gameplay or perform neural inference.

## Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `inference` extra reserves ONNX Runtime for later. The separate `training` extra is optional and installs Ultralytics; neither is needed for Phase 1. Do not redistribute weights or game imagery without checking their licenses and rights.

## Commands

```powershell
game-vision windows list
game-vision capture --hwnd 0x123456 --count 20 --interval 2 --out .\local-data
game-vision replay --frames .\local-data
game-vision benchmark --source synthetic --duration 30 --out .\benchmark-report.json
python -m pytest
```

Window listing prints hexadecimal HWND, visible client dimensions, physical screen origin, state, and title. Capture is explicit and saves only when this command is requested. Re-querying the selected HWND handles movement and resize; a missing or minimized target stops capture without selecting another window. A screen crop sees the currently visible pixels: it cannot reliably recover minimized content and may include occlusion when another window covers the target.

`replay` reads PNG/JPEG images in sorted filename order. Synthetic source and benchmark work offline. Benchmark JSON includes process RSS and a CPU sample (psutil), capture/change processing latency distributions, FPS, and unchanged skip rate; startup is not represented as steady-state performance.

## Safety, privacy, and scope

No app runtime network access, game memory access, private APIs, or input injection is implemented. Frames are not persisted by the reusable capture source. The capture CLI writes images to the chosen local directory; `.gitignore` excludes local datasets and image files. Inspect destinations before sharing or committing. Game terms of service may restrict observation or data collection; users are responsible for checking applicable terms. There is no anti-cheat interaction or bypass.

## Limitations

Windows enumeration/capture needs Windows 10/11 and an interactive desktop. DPI-awareness setup and client-to-screen mapping are best effort across mixed-DPI configurations and need manual validation on those machines. MSS captures desktop pixels, so occlusion and minimized windows are inherently limited. Change detection uses sampled pixel differences, not semantic understanding. YOLO26n/ONNX, OCR, annotation, learning, persistence limits, and simulator execution are future work.

See [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md), and [CONTRIBUTING.md](CONTRIBUTING.md). Project license: AGPL-3.0-only; third-party components retain their own licenses (NumPy BSD-3-Clause, MSS MIT, Pillow HPND, psutil BSD-3-Clause, pytest MIT, Ruff MIT, Ultralytics AGPL-3.0). No models, game assets, or screenshots are bundled.
