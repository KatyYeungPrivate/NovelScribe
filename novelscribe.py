import argparse
import bisect
import glob
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
import paddle
from PIL import Image
from paddleocr import PaddleOCR

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^黑幫少爺愛上我\s*第一部\s*上$"),
    re.compile(r"^本章第\s*\d+\s*頁\s*[\/／]\s*共\s*\d+\s*頁$"),
    re.compile(r"^#\d+$"),
    re.compile(r"^\d{1,3}%$"),
    re.compile(r"^[<>\\[\\]◀▶]$"),
]

# CJK unified ideographs, CJK punctuation, fullwidth forms
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")
GAP_MULTIPLIER = 1.5
MIN_NON_CJK_CHARS = 5

DIVIDER_ICON_PATH = "gun_icon.png"
DIVIDER_MATCH_THRESHOLD = 0.70
DIVIDER_LOCAL_STD_THRESHOLD = 10.0


def should_remove(text: str) -> bool:
    s = text.strip()
    if any(p.fullmatch(s) for p in HEADER_FOOTER_PATTERNS):
        return True
    # Keep short digit-only strings (e.g. chapter numbers like "0")
    if s.isdigit() and len(s) <= 2:
        return False
    # Drop short stray UI characters that are not Chinese/punctuation
    if len(s) <= 2 and not _CJK_RE.search(s):
        return True
    return False


def sort_lines(texts, boxes):
    items = []
    for t, b in zip(texts, boxes):
        b = [int(v) for v in b]
        items.append((t, b))
    items.sort(key=lambda x: (x[1][1], x[1][0]))
    return items


def _is_bar_char(ch):
    return ch in "|｜"


def _box_overlaps_as_bar(box, other, h_margin=30, v_margin=5):
    """True if `box` looks like a vertical bar next to/through `other`."""
    horizontal_close = not (box[2] + h_margin < other[0] or box[0] - h_margin > other[2])
    vertical_close = not (box[3] + v_margin < other[1] or box[1] - v_margin > other[3])
    return horizontal_close and vertical_close


def cleanup_ocr_items(items):
    """Remove decorative bars that OCR misread as digits or pipes.

    Keeps chapter numbers that sit above/below the title (no vertical overlap
    with the CJK text) while removing side bars that run through/next to it.
    """
    # Strip leading/trailing bar chars from text.
    cleaned = []
    for text, box in items:
        while text and _is_bar_char(text[0]):
            text = text[1:].strip()
        while text and _is_bar_char(text[-1]):
            text = text[:-1].strip()
        if text:
            cleaned.append((text, box))

    # Drop stray single-char digits/pipes that sit right next to CJK text.
    final = []
    for i, (text, box) in enumerate(cleaned):
        s = text.strip()
        if len(s) == 1 and (s.isdigit() or _is_bar_char(s)):
            near_cjk = False
            for j, (other_text, other_box) in enumerate(cleaned):
                if i == j:
                    continue
                if not _CJK_RE.search(other_text):
                    continue
                if _box_overlaps_as_bar(box, other_box):
                    near_cjk = True
                    break
            if near_cjk:
                continue
        final.append((text, box))
    return final


def is_empty_page(filtered_lines):
    total = sum(len(line) for line in filtered_lines)
    cjk_count = sum(len(_CJK_RE.findall(line)) for line in filtered_lines)
    # Skip pages that have no CJK text and very few other characters
    # (e.g. illustration pages with "LOVE" or a stray symbol).
    return cjk_count == 0 and total <= MIN_NON_CJK_CHARS


def detect_paragraph_dividers(image_path, template_path, items, page_width):
    if not os.path.exists(template_path):
        return []
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if img is None or tmpl is None:
        return []

    h, w = img.shape
    th, tw = tmpl.shape
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    idx = np.unravel_index(np.argsort(res, axis=None)[::-1], res.shape)
    used = np.zeros(res.shape, dtype=bool)
    sy = max(1, th // 2)
    sx = max(1, tw // 2)

    text_boxes = [box for _, box in items]
    centers = []

    for y, x in zip(*idx):
        score = res[y, x]
        if score < DIVIDER_MATCH_THRESHOLD:
            break
        if used[y, x]:
            continue

        cx, cy = x + tw / 2, y + th / 2

        # Ignore matches too far from the horizontal center.
        if not (page_width * 0.25 <= cx <= page_width * 0.75):
            continue
        # Ignore matches right at the page edges.
        if cy < 20 or cy > h - 20:
            continue
        # Ignore matches that sit inside an OCR text box.
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in text_boxes):
            continue

        # Local background uniformity check: the icon should be on a
        # plain background, not embedded in an illustration.
        pad = max(tw, th) // 2
        x0 = int(max(0, cx - tw / 2 - pad))
        y0 = int(max(0, cy - th / 2 - pad))
        x1 = int(min(w, cx + tw / 2 + pad))
        y1 = int(min(h, cy + th / 2 + pad))
        win = img[y0:y1, x0:x1].astype(np.float32)

        cx0, cy0 = int(cx - tw / 2), int(cy - th / 2)
        cx1, cy1 = int(cx + tw / 2), int(cy + th / 2)
        mask = np.ones(win.shape, dtype=bool)
        mask[
            max(0, cy0 - y0) : min(win.shape[0], cy1 - y0),
            max(0, cx0 - x0) : min(win.shape[1], cx1 - x0),
        ] = False

        if mask.sum() < 10:
            continue
        outside_std = np.std(win[mask])
        if outside_std >= DIVIDER_LOCAL_STD_THRESHOLD:
            continue

        centers.append(int(cy))
        used[max(0, y - sy) : y + sy, max(0, x - sx) : x + sx] = True

    return sorted(centers)


def is_centered_page(items, page_width):
    if page_width <= 0 or not items:
        return False
    mean_x = statistics.mean(box[0] for _, box in items)
    # Centered text sits past the first third of the page.
    return mean_x > page_width * 0.30


def rebuild_page(items, page_width=0):
    if not items:
        return []

    # Centered pages (title/copyright) should keep each line on its own line.
    if is_centered_page(items, page_width):
        return [(text, box[1]) for text, box in items]

    heights = [box[3] - box[1] for _, box in items]
    med_h = statistics.median(heights) if heights else 0

    # Find the leftmost repeated left edge (body-left) rounded to 10 px.
    rounded = [int(round(box[0] / 10.0)) * 10 for _, box in items]
    counter = Counter(rounded)
    repeated = [k for k, v in counter.items() if v > 1]
    body_left = min(repeated) if repeated else min(counter)

    # A paragraph-start line is shifted right by ~2 full-width characters.
    indent_threshold = body_left + max(20, int(med_h * 0.8))
    gap_threshold = med_h * GAP_MULTIPLIER

    paragraphs = []
    current = []
    current_y = None
    prev_box = None
    for text, box in items:
        is_new = False
        if not current:
            is_new = True
        else:
            if box[0] >= indent_threshold:
                is_new = True
            elif prev_box and box[1] - prev_box[3] > gap_threshold:
                is_new = True

        if is_new:
            if current:
                paragraphs.append(("".join(current), current_y))
            current = ["　　" + text]
            current_y = box[1]
        else:
            current.append(text)
        prev_box = box

    if current:
        paragraphs.append(("".join(current), current_y))
    return paragraphs


def process_image(ocr_engine, path, template_path=DIVIDER_ICON_PATH):
    results = ocr_engine.predict(str(path))
    if not results:
        return []
    res = results[0]
    texts = list(res.get("rec_texts", []))
    boxes = list(res.get("rec_boxes", []))

    items = sort_lines(texts, boxes)
    filtered = [(t, b) for t, b in items if not should_remove(t)]
    filtered = cleanup_ocr_items(filtered)

    if is_empty_page([t for t, _ in filtered]):
        return []

    with Image.open(path) as img:
        page_width = img.width

    page_tuples = rebuild_page(filtered, page_width)

    # Only insert paragraph-divider markers on body-text pages.
    if page_tuples and not is_centered_page(filtered, page_width):
        icon_ys = detect_paragraph_dividers(path, template_path, filtered, page_width)
        ys = [line_y for _, line_y in page_tuples]
        for y in icon_ys:
            idx = bisect.bisect_left(ys, y)
            page_tuples.insert(idx, ("---", y))
            ys.insert(idx, y)

    return [text for text, _ in page_tuples]


def main():
    parser = argparse.ArgumentParser(
        description="NovelScribe: batch OCR for Traditional Chinese novel screenshots"
    )
    parser.add_argument("input_dir", nargs="?", default="TestImage",
                        help="Directory containing the screenshot images")
    parser.add_argument("output_file", nargs="?", default="output\\combined.txt",
                        help="Combined text output path")
    parser.add_argument(
        "--divider-icon",
        default=DIVIDER_ICON_PATH,
        help="Path to a small paragraph-divider icon template (PNG). If not found, no divider detection is performed.",
    )
    args = parser.parse_args()

    image_paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*.png")),
        key=lambda p: int(re.search(r"\((\d+)\)", Path(p).name).group(1)),
    )
    if not image_paths:
        print(f"No PNG images found in {args.input_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "gpu:0" if paddle.is_compiled_with_cuda() and paddle.cuda.device_count() > 0 else "cpu"
    print(f"Initializing PaddleOCR (lang=chinese_cht, device={device}) ...")
    ocr = PaddleOCR(
        lang="chinese_cht",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        prev_centered = None
        for i, img_path in enumerate(image_paths, 1):
            name = Path(img_path).name
            print(f"[{i}/{len(image_paths)}] {name}", flush=True)
            page_lines = process_image(ocr, img_path, template_path=args.divider_icon)
            if not page_lines:
                continue

            is_centered = not any(line.startswith("　　") for line in page_lines)

            if prev_centered is not None:
                if not prev_centered and is_centered:
                    # After body text and before a centered/empty/graphical page: 5 empty lines.
                    f.write("\n\n\n\n\n\n")
                elif prev_centered or is_centered:
                    # Centered pages get 3 empty lines of separation.
                    f.write("\n\n\n\n")
                else:
                    # Body pages get 1 empty line of separation.
                    f.write("\n\n")

            if is_centered:
                f.write("\n".join(page_lines))
            else:
                f.write("\n\n".join(page_lines))

            prev_centered = is_centered

    print(f"\nDone. Combined output written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
