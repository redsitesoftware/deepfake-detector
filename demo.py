from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from deepfake_detector import Detector


def _parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _draw_confidence_bar(frame: np.ndarray, confidence: float, origin: tuple[int, int]) -> None:
    x, y = origin
    width = 180
    height = 16
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    fill_width = int(width * max(0.0, min(confidence, 1.0)))
    bar_color = (0, 255, 0) if confidence <= 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), bar_color, -1)
    cv2.putText(
        frame,
        f"confidence {confidence:.2f}",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_overlay(frame: np.ndarray, result, fps: float) -> np.ndarray:
    output = frame.copy()
    color = (0, 0, 255) if result.is_fake else (0, 255, 0)

    if result.face_bbox is not None:
        x, y, w, h = result.face_bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    status_text = "FAKE" if result.is_fake else "REAL"
    cv2.putText(
        output,
        f"{status_text} {result.confidence:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    _draw_confidence_bar(output, result.confidence, (20, 50))

    signal_lines = [
        f"CNN: {result.signals['cnn']:.2f}" if result.signals['cnn'] is not None else "CNN: --",
        (
            f"Temporal: {result.signals['temporal']:.2f}"
            if result.signals['temporal'] is not None
            else "Temporal: --"
        ),
        (
            f"Liveness: {result.signals['liveness']:.2f}"
            if result.signals['liveness'] is not None
            else "Liveness: --"
        ),
        f"FPS: {fps:.1f}",
    ]
    for index, line in enumerate(signal_lines, start=0):
        cv2.putText(
            output,
            line,
            (20, 95 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deepfake detector demo")
    parser.add_argument("--source", required=True, help="Webcam index or video file path")
    parser.add_argument("--output", help="Optional path to save annotated output video")
    args = parser.parse_args()

    source = _parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    detector = Detector(fps=fps)
    frame_count = 0
    started_at = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_count += 1
            result = detector.analyse(frame)
            elapsed = max(time.perf_counter() - started_at, 1e-6)
            measured_fps = frame_count / elapsed
            overlay = _draw_overlay(frame, result, measured_fps)

            cv2.imshow("deepfake-detector demo", overlay)
            if writer is not None:
                writer.write(overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
