"""Write the synthetic README images. Uses documentation addresses only."""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "images"


def frame() -> np.ndarray:
    image = np.full((220, 640, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (616, 196), (36, 36, 36), -1)
    cv2.putText(
        image,
        "Server 203.0.113.10",
        (48, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "ada@example.com",
        (48, 156),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return image


def pixelate(image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    crop = image[y : y + h, x : x + w]
    small = cv2.resize(crop, (8, 4), interpolation=cv2.INTER_LINEAR)
    image[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = frame()
    after = before.copy()
    pixelate(after, 150, 78, 250, 42)
    pixelate(after, 40, 132, 250, 36)
    cv2.imwrite(str(OUT / "synthetic-before.png"), before)
    cv2.imwrite(str(OUT / "synthetic-after.png"), after)


if __name__ == "__main__":
    main()
