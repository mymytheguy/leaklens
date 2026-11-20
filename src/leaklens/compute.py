"""Select ONNX Runtime compute backend. Never fail if GPU is missing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_compute(*, force_cpu: bool = False) -> tuple[str, dict]:
    """Return (label, RapidOCR param overrides)."""
    if force_cpu:
        return "CPU", {}
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception as exc:
        logger.info("onnxruntime provider query failed: %s", exc)
        return "CPU", {}

    if "CUDAExecutionProvider" in providers:
        return "NVIDIA GPU", {"EngineConfig.onnxruntime.use_cuda": True}
    if "TensorrtExecutionProvider" in providers:
        return "NVIDIA GPU", {"EngineConfig.onnxruntime.use_cuda": True}
    if "DmlExecutionProvider" in providers:
        return "GPU (DirectML)", {"EngineConfig.onnxruntime.use_dml": True}
    if "CoreMLExecutionProvider" in providers:
        return "Apple GPU", {"EngineConfig.onnxruntime.use_coreml": True}
    return "CPU", {}
