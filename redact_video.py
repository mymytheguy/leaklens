#!/usr/bin/env python3
"""Backwards-compatible directory-based entry point.

Prefer the packaged CLI:

    leaklens input.mp4
"""

from __future__ import annotations

import sys

from leaklens.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main())
