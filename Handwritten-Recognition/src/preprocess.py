"""
Image Preprocessing and Multi-Character Segmentation Module
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass


def process_single_crop(crop_img: np.ndarray, thicken_stroke: bool = True) -> tuple[np.ndarray, Image.Image]:
    """
    Takes a single binary cropped character region, resizes it into a 20x20 box,
    pads onto a 28x28 canvas, aligns via Center-of-Mass, and normalizes to [0, 1].
    """
    h, w = crop_img.shape[:2]
    if h == 0 or w == 0:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank.reshape(1, 28, 28, 1), Image.fromarray((blank * 255).astype(np.uint8))

    # Morphological dilation to thicken thin strokes
    if thicken_stroke:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        crop_img = cv2.dilate(crop_img, kernel, iterations=1)

    # Fit into 20x20 box preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(int(round((h * 20.0) / w)), 1)
    else:
        new_h = 20
        new_w = max(int(round((w * 20.0) / h)), 1)

    resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad onto 28x28 black canvas
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    canvas_28x28 = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    # Center of Mass adjustment to (14, 14)
    cy, cx = center_of_mass(canvas_28x28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas_28x28 = cv2.warpAffine(
            canvas_28x28,
            M,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    normalized = canvas_28x28.astype("float32") / 255.0
    tensor = normalized.reshape(1, 28, 28, 1)
    preview = Image.fromarray(canvas_28x28)

    return tensor, preview


def preprocess_image(image, thicken_stroke: bool = True) -> tuple[np.ndarray, Image.Image]:
    """Single image preprocessing fallback / standard pipeline."""
    # Convert input to grayscale numpy array
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            canvas_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas_bg, image)
        gray = np.array(image.convert("L"), dtype=np.uint8)
    else:
        gray = np.array(image, dtype=np.uint8)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank.reshape(1, 28, 28, 1), Image.fromarray((blank * 255).astype(np.uint8))

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y + h, x:x + w]
    return process_single_crop(cropped, thicken_stroke=thicken_stroke)


def segment_and_preprocess_characters(image, thicken_stroke: bool = True) -> list[dict]:
    """
    Intelligently detects and segments one or multiple characters/digits from left to right.
    Returns a list of dicts:
    [
        {"tensor": (1, 28, 28, 1), "preview": Image, "bbox": (x, y, w, h)},
        ...
    ]
    """
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            canvas_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas_bg, image)
        gray = np.array(image.convert("L"), dtype=np.uint8)
    else:
        gray = np.array(image, dtype=np.uint8)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes = []
    min_area = 150  # Filter out tiny accidental dots / edge clicks
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h >= min_area or (w >= 10 and h >= 10)) and w >= 6 and h >= 10:
            valid_boxes.append((x, y, w, h))

    # If no contours found, fallback to whole image
    if not valid_boxes:
        coords = cv2.findNonZero(thresh)
        if coords is None:
            tensor, preview = process_single_crop(thresh, thicken_stroke=thicken_stroke)
            return [{"tensor": tensor, "preview": preview, "bbox": (0, 0, 28, 28)}]
        valid_boxes = [cv2.boundingRect(coords)]

    # Sort from left to right
    valid_boxes = sorted(valid_boxes, key=lambda b: b[0])

    # Merge overlapping or vertically stacked parts of same character (e.g. dot on 'i' or 'j', or equals signs)
    merged_boxes = []
    for box in valid_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue
        prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
        curr_x, curr_y, curr_w, curr_h = box
        # Check horizontal overlap
        overlap_x = max(0, min(prev_x + prev_w, curr_x + curr_w) - max(prev_x, curr_x))
        if overlap_x > 0.4 * min(prev_w, curr_w):
            # Merge into one bounding box
            nx = min(prev_x, curr_x)
            ny = min(prev_y, curr_y)
            nw = max(prev_x + prev_w, curr_x + curr_w) - nx
            nh = max(prev_y + prev_h, curr_y + curr_h) - ny
            merged_boxes[-1] = (nx, ny, nw, nh)
        else:
            merged_boxes.append(box)

    results = []
    for (x, y, w, h) in merged_boxes:
        crop = thresh[y:y + h, x:x + w]
        tensor, preview = process_single_crop(crop, thicken_stroke=thicken_stroke)
        results.append({
            "tensor": tensor,
            "preview": preview,
            "bbox": (x, y, w, h)
        })

    return results
