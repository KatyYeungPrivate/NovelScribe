import argparse
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
SPACING_CHAR_THRESHOLD = 10
GAP_MULTIPLIER = 1.5


def should_remove(text: str) -> bool:
    s = text.strip()
    if any(p.fullmatch(s) for p in HEADER_FOOTER_PATTERNS):
        return True
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


def is_spacing_page(filtered_lines):
    total = sum(len(line) for line in filtered_lines)
    return total <= SPACING_CHAR_THRESHOLD


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
        return [text for text, _ in items]

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
                paragraphs.append("".join(current))
            current = ["　　" + text]
        else:
            current.append(text)
        prev_box = box

    if current:
        paragraphs.append("".join(current))
    return paragraphs


def process_image(ocr_engine, path):
    results = ocr_engine.predict(str(path))
    if not results:
        return []
    res = results[0]
    texts = list(res.get("rec_texts", []))
    boxes = list(res.get("rec_boxes", []))

    items = sort_lines(texts, boxes)
    filtered = [(t, b) for t, b in items if not should_remove(t)]

    if is_spacing_page([t for t, _ in filtered]):
        return ["---"]

    with Image.open(path) as img:
        page_width = img.width
    return rebuild_page(filtered, page_width)


def main():
    parser = argparse.ArgumentParser(
        description="NovelScribe: batch OCR for Traditional Chinese novel screenshots"
    )
    parser.add_argument("input_dir", nargs="?", default="TestImage",
                        help="Directory containing the screenshot images")
    parser.add_argument("output_file", nargs="?", default="output\\combined.txt",
                        help="Combined text output path")
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
        prev_was_divider = False
        for i, img_path in enumerate(image_paths, 1):
            name = Path(img_path).name
            print(f"[{i}/{len(image_paths)}] {name}", flush=True)
            page_lines = process_image(ocr, img_path)
            is_divider = page_lines == ["---"]
            if is_divider and prev_was_divider:
                continue
            f.write("\n".join(page_lines))
            f.write("\n\n")
            prev_was_divider = is_divider

    print(f"\nDone. Combined output written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
