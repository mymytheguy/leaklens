"""Typed errors mapped to process exit codes."""

from __future__ import annotations


class LeakLensError(Exception):
    """Generic processing error (exit 1)."""

    exit_code = 1


class UsageError(LeakLensError):
    """Invalid arguments or missing input (exit 2)."""

    exit_code = 2


class UnsupportedMediaError(LeakLensError):
    """File exists but is not a readable video (exit 3)."""

    exit_code = 3


class PipelineError(LeakLensError):
    """Detection pipeline failure (exit 4)."""

    exit_code = 4


class ExportError(LeakLensError):
    """Failed to write redacted video or reports (exit 5)."""

    exit_code = 5
