"""RGB-only red/white shelf marker proposals and Tesseract text recognition.

No catalog, simulator labels, depth, or world coordinates are used by detection.
This deliberately supports the installed sign design, not arbitrary warehouse OCR.
"""

import csv
import io
import re
import shutil
import subprocess

import cv2
import numpy as np


class ShelfMarkerReader:
    def __init__(self, *, host_tesseract=False):
        self.command = (
            ["nsenter", "-t", "1", "-m", "--", "/usr/bin/tesseract"]
            if host_tesseract
            else ["tesseract"]
        )
        if not shutil.which(self.command[0]):
            raise RuntimeError(
                "Tesseract is required (or --host-tesseract on this shared host)"
            )
        self.version = subprocess.run(
            self.command + ["--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0]

    def text(self, binary, whitelist, psm=7):
        # PNG travels via stdin; no shell, no shared temporary file names.
        padded = cv2.copyMakeBorder(
            binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
        )
        data = cv2.imencode(".png", padded)[1].tobytes()
        result = subprocess.run(
            self.command
            + [
                "stdin",
                "stdout",
                "--psm",
                str(psm),
                "-c",
                "tessedit_char_whitelist=" + whitelist,
                "tsv",
            ],
            input=data,
            capture_output=True,
            check=True,
            timeout=10,
        )
        rows = list(csv.DictReader(io.StringIO(result.stdout.decode()), delimiter="\t"))
        words = [r for r in rows if r.get("text", "").strip()]
        text = "".join(r["text"].strip() for r in words)
        confidence = min((float(r["conf"]) for r in words), default=-1)
        return text, confidence

    def digits(self, binary):
        """Read two separated glyphs, avoiding OCR's whole-word substitutions."""
        contours, _ = cv2.findContours(
            255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = sorted(
            (x, y, w, h)
            for x, y, w, h in map(cv2.boundingRect, contours)
            if h > binary.shape[0] * 0.35 and w > 8
        )
        if len(boxes) != 2:
            return "", -1.0, []
        reads = []
        for x, y, w, h in boxes:
            glyph = cv2.resize(
                binary[y : y + h, x : x + w], (max(1, round(w * 100 / h)), 100)
            )
            reads.append(self.text(glyph, "0123456789OA", psm=10))
        if any(not re.fullmatch(r"[0-9OA]", text) for text, _ in reads):
            return "", -1.0, reads
        return (
            "".join(
                text.translate(str.maketrans({"O": "0", "A": "4"})) for text, _ in reads
            ),
            min(conf for _, conf in reads),
            reads,
        )

    def detect(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red = (
            ((hsv[..., 0] < 4) | (hsv[..., 0] > 170))
            & (hsv[..., 1] > 110)
            & (hsv[..., 2] > 65)
        ).astype(np.uint8) * 255
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 250:
                continue
            quad = cv2.approxPolyDP(
                contour, 0.025 * cv2.arcLength(contour, True), True
            ).reshape(-1, 2)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            x, y, w, h = cv2.boundingRect(quad)
            if min(w, h) < 18 or not 0.45 < w / h < 1.4:
                continue
            if x < 2 or y < 2 or x + w >= rgb.shape[1] - 2 or y + h >= rgb.shape[0] - 2:
                continue
            # Order clockwise: top-left, top-right, bottom-right, bottom-left.
            sums, diffs = quad.sum(1), np.diff(quad, axis=1).ravel()
            points = np.float32(
                [
                    quad[sums.argmin()],
                    quad[diffs.argmin()],
                    quad[sums.argmax()],
                    quad[diffs.argmax()],
                ]
            )
            if len(np.unique(points, axis=0)) != 4:
                continue
            points[0] -= (points[1] - points[0]) * (580 / 444)
            points[3] -= (points[2] - points[3]) * (580 / 444)
            x, y, w, h = cv2.boundingRect(points)
            if x < 2 or y < 2 or x + w >= rgb.shape[1] - 2 or y + h >= rgb.shape[0] - 2:
                continue
            tf = cv2.getPerspectiveTransform(
                points, np.float32([[0, 0], [1023, 0], [1023, 511], [0, 511]])
            )
            patch = cv2.warpPerspective(rgb, tf, (1024, 512))
            phsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            pred = ((phsv[..., 0] < 12) | (phsv[..., 0] > 165)) & (phsv[..., 1] > 100)
            # Layout check rejects solid red inventory boxes before invoking OCR.
            if (
                not 0.05 < pred[40:-40, 30:550].mean() < 0.5
                or pred[40:-40, 620:-30].mean() < 0.55
            ):
                continue
            left = np.where(pred[55:440, 35:540], 0, 255).astype(np.uint8)
            right = np.where(pred[85:425, 610:990], 255, 0).astype(np.uint8)
            number, nc, raw_digits = self.digits(left)
            suffix, sc = self.text(right, "/0123456789RS")
            # At the numeric row position, OCR's S glyph denotes the digit 5.
            # Preserve the raw read in provenance; never fill missing characters.
            raw_suffix = suffix
            suffix = suffix.replace("S", "5")
            if not re.fullmatch(r"\d{2}", number) or not re.fullmatch(
                r"/?[1-9]R", suffix
            ):
                continue
            # Tesseract sometimes assigns zero confidence with a character whitelist.
            # These are observations, not trusted identities: temporal+geometric
            # consistency and independent evaluation remain necessary.
            label = number + "/" + suffix.lstrip("/")
            detections.append(
                {
                    "label": label,
                    "roi": [x, y, x + w, y + h],
                    "quad": points.tolist(),
                    "ocr_confidence": [nc, sc],
                    "raw_ocr": {"digits": raw_digits, "suffix": raw_suffix},
                    "rectified": patch,
                }
            )
        return sorted(detections, key=lambda d: d["roi"])


def planar_marker_mask(mask, depth, K, tolerance=0.02):
    """Keep the dominant depth plane of a segmented rigid placard (RGB-D only)."""
    valid = np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    if len(xx) < 30:
        return None
    K = np.asarray(K)
    z = depth[yy, xx]
    xyz = np.c_[(xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z]
    rng = np.random.default_rng(0)
    sample = xyz[rng.choice(len(xyz), min(600, len(xyz)), replace=False)]
    best = None
    best_count = 0
    for _ in range(80):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        normal /= length
        inliers = np.abs((sample - a) @ normal) < tolerance
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None or best_count < 0.65 * len(sample):
        return None
    center = sample[best].mean(0)
    _, _, vt = np.linalg.svd(sample[best] - center, full_matrices=False)
    keep = np.abs((xyz - center) @ vt[-1]) < tolerance
    refined = np.zeros_like(valid)
    refined[yy[keep], xx[keep]] = True
    return refined
