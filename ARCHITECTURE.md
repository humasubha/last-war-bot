# Architecture

`FrameSource` returns RGB frames with identity, timestamp, and optional desktop geometry. `WindowLocator` provides HWND-based top-level discovery. The Windows source rechecks the selected HWND and client bounds on each read, then uses MSS to crop those screen coordinates. It does not follow a replacement window.

`ChangeDetector` compares downsampled frames and returns mean absolute magnitude, changed fraction, and a bounding region. ROI and ignored regions are configurable. `ObjectDetector` returns typed detections with pixel and normalized boxes, center, frame identity, and time. `Yolo26OnnxAdapter` is deliberately unimplemented and raises instead of fabricating results.

`Observation` and `GameState` separate perception from state. `ActionProposal` carries priority, confidence, preconditions, expected outcome, timeout, and source-frame identity. `ActionOutcome` reserves observed result and recovery fields. `NoOpExecutor` never sends inputs, including when passed a simulator flag. A later simulator must independently implement stale-observation rejection, ambiguity/confidence handling, retry limits, and outcome evaluation; desktop/live input stays out of scope.

The pipeline can place cheap change detection before inference; inference scheduling, cached detections, adaptive rates, and timing contracts are reserved. Dataset collection metadata, retention/quotas, scene diversity, annotation, YOLO export and training are not implemented in this milestone.

Runtime dependencies are NumPy, MSS, Pillow, and psutil. Training is an optional extra separated from production inference. Ultralytics is AGPL-3.0; distribution with its code may impose corresponding obligations. This repository selects AGPL-3.0-or-later for the project and does not bundle Ultralytics, ONNX models, or game assets. Review upstream license texts before redistribution.
