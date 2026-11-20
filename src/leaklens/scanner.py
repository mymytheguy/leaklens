"""High-level scan API. Detection logic stays in detectors, tracking, and pipeline."""

from __future__ import annotations

from leaklens.pipeline import ScanResult, run_pipeline, run_scan

__all__ = ["ScanResult", "run_pipeline", "run_scan"]
