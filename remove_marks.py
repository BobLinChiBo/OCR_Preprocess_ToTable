#!/usr/bin/env python3
"""
remove_marks.py
---------------------
Remove everything except text/table lines by:

1. Otsu threshold → binary foreground (text + lines).
2. Mild dilation to make sure every stroke is covered.
3. Copy original gray pixels where the mask is white; paint the rest white.

USAGE
-----
python remove_marks.py input.jpg output.png
python remove_marks.py input.jpg output.png --dilate_iter 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_protect_mask(gray: np.ndarray, dilate_iter: int) -> np.ndarray:
    """Return a solid white mask for every text / line pixel."""
    # Otsu → dark foreground becomes white (255)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Safety dilation so no thin strokes fall through
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    protect = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return protect


def process_page(path_in: Path, path_out: Path, dilate_iter: int = 2):
    """Main cleaning pipeline."""
    bgr = cv2.imread(str(path_in))
    if bgr is None:
        raise FileNotFoundError(path_in)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    protect = build_protect_mask(gray, dilate_iter)

    # Produce final: keep gray under protect, paint rest white
    clean = np.full_like(gray, 255)
    clean[protect > 0] = gray[protect > 0]

    # Save as PNG (lossless, 8-bit grayscale)
    cv2.imwrite(str(path_out), clean)


def main():
    ap = argparse.ArgumentParser(
        description="Strip non-text/non-line artefacts by flipping a protect mask"
    )
    ap.add_argument("input", type=Path, help="input image")
    ap.add_argument("output", type=Path, help="output cleaned image")
    ap.add_argument(
        "--dilate_iter",
        type=int,
        default=2,
        help="extra dilation iterations to enlarge the protect mask (default 2)",
    )
    args = ap.parse_args()

    process_page(args.input, args.output, args.dilate_iter)


if __name__ == "__main__":
    main()
