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

# CJK unified ideographs, CJK punctuation, fullwidth forms
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")
GAP_MULTIPLIER = 1.5
MIN_NON_CJK_CHARS = 5

DIVIDER_ICON_PATH = "gun_icon.png"
DIVIDER_MATCH_THRESHOLD = 0.70
DIVIDER_LOCAL_STD_THRESHOLD = 10.0


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


def ensure_em_dash_spacing(text, min_blank=3):
    """Ensure lines beginning with an em dash have at least `min_blank` empty lines above."""
    lines = text.splitlines()
    result = []
    for line in lines:
        stripped = line.lstrip(" \t\u3000")
        if stripped and stripped[0] in "\u2014\u2015":
            blanks = 0
            while blanks < len(result) and result[-(blanks + 1)] == "":
                blanks += 1
            if result and blanks < min_blank:
                result.extend([""] * (min_blank - blanks))
        result.append(line)
    return "\n".join(result)


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


def filter_header_footer_content(items):
    """Filter out header/footer content based on text patterns."""
    filtered = []
    for text, box in items:
        # Skip lines that look like header/footer content
        # Patterns: page numbers, percentages, copyright symbols, short titles
        text_stripped = text.strip()
        
        # Skip if it's just a number or percentage
        if text_stripped.replace('%', '').replace('#', '').replace('©', '').replace('／', '').replace('/', '').replace(' ', '').replace('　', '').isdigit():
            continue
        
        # Skip if it's a page number pattern like "本章第11頁／共11頁"
        if '本章第' in text_stripped and '頁' in text_stripped:
            continue
        
        # Skip if it's very short and looks like navigation (e.g., "#449", "25%")
        if len(text_stripped) < 10 and any(c in text_stripped for c in '#%©'):
            continue
        
        # Skip if it's the book title appearing alone
        if text_stripped == '黑幫少爺愛上我第二部上':
            continue
        
        filtered.append((text, box))
    
    return filtered


def _body_left_and_threshold(items):
    """Return (body_left, indent_threshold, med_h)."""
    heights = [box[3] - box[1] for _, box in items]
    med_h = statistics.median(heights) if heights else 0

    # Body-left is the leftmost x that is part of the main text column.
    # We find the smallest x that has a neighbor within a small band, which
    # keeps stray page numbers/footers from being chosen while still allowing
    # a single wrapped line at the true margin to count.
    x_positions = sorted({box[0] for _, box in items})
    cluster_width = max(80, int(med_h * 2.5))
    body_left = None
    for i, x in enumerate(x_positions):
        for other in x_positions[i + 1:]:
            if other - x <= cluster_width:
                body_left = x
                break
        if body_left is not None:
            break
    if body_left is None:
        body_left = min(box[0] for _, box in items)

    # A paragraph-start line is shifted right by at least ~1/3 full-width character.
    indent_threshold = body_left + max(10, int(med_h * 0.25))
    return body_left, indent_threshold, med_h


def rebuild_page(items, page_width=0):
    if not items:
        return []

    # Centered pages (title/copyright) should keep each line on its own line.
    if is_centered_page(items, page_width):
        return [(text, box[1]) for text, box in items]

    body_left, indent_threshold, med_h = _body_left_and_threshold(items)
    gap_threshold = med_h * GAP_MULTIPLIER
    
    # Check if the first line is a continuation (starts at body-left or with whitespace)
    first_is_continuation = False
    if items:
        first_text, first_box = items[0]
        first_starts_with_whitespace = first_text.startswith((' ', '　', '\t'))
        first_is_continuation = (first_box[0] < indent_threshold + 2) or first_starts_with_whitespace

    paragraphs = []
    current = []
    current_y = None
    prev_box = None
    for text, box in items:
        is_new = False
        if not current:
            is_new = True
        else:
            # Check if this line starts with whitespace (continuation)
            starts_with_whitespace = text.startswith((' ', '　', '\t'))
            
            # Special handling: if the first line was a continuation, be more lenient with paragraph detection
            # to avoid splitting continuation lines that are at the same indentation
            if first_is_continuation:
                # Only treat as new paragraph if:
                # 1. It's significantly indented (more than threshold + margin) AND doesn't start with whitespace, OR
                # 2. There's a large vertical gap
                if box[0] >= indent_threshold + 10 and not starts_with_whitespace:
                    is_new = True
                elif prev_box and box[1] - prev_box[3] > gap_threshold:
                    is_new = True
            else:
                # Normal paragraph detection
                if box[0] >= indent_threshold and not starts_with_whitespace:
                    is_new = True
                elif prev_box and box[1] - prev_box[3] > gap_threshold:
                    is_new = True

        if is_new:
            if current:
                paragraphs.append(("".join(current), current_y))
            current = ["　　" + text]
            current_y = box[1]
        else:
            # If line starts with whitespace, add it without the leading whitespace
            if text.startswith((' ', '　', '\t')):
                current.append(text.lstrip(' 　\t'))
            else:
                current.append(text)
        prev_box = box

    if current:
        paragraphs.append(("".join(current), current_y))
    return paragraphs


def detect_content_region(ocr_engine, image_paths, sample_count=5):
    """
    Detect header/footer regions and side margins by sampling pages and finding where
    repeated text appears at edges.

    Returns: (top_margin, bottom_margin, left_margin, right_margin) - pixels to exclude from edges
    """
    if len(image_paths) < 2:
        return 0, 0, 0, 0  # No margin if can't detect

    # Sample evenly distributed pages
    sample_indices = [int(i * len(image_paths) / sample_count)
                      for i in range(sample_count)]
    sample_paths = [image_paths[i] for i in sample_indices]

    # Collect text positions from sample pages
    all_y_positions = []  # List of (y_center, page_height)
    all_x_positions = []  # List of (x_center, page_width)

    for path in sample_paths:
        results = ocr_engine.predict(str(path))
        if not results:
            continue
        res = results[0]
        texts = list(res.get("rec_texts", []))
        boxes = list(res.get("rec_boxes", []))

        with Image.open(path) as img:
            page_height = img.height
            page_width = img.width

        for text, box in zip(texts, boxes):
            text = text.strip()
            if text:  # Skip empty
                y_center = (box[1] + box[3]) / 2
                x_center = (box[0] + box[2]) / 2
                all_y_positions.append((y_center, page_height))
                all_x_positions.append((x_center, page_width))

    if not all_y_positions:
        return 0, 0, 0, 0

    # Normalize positions as percentage of page dimensions
    normalized_y_positions = [y / h for y, h in all_y_positions]
    normalized_x_positions = [x / w for x, w in all_x_positions]

    # Find header region: text consistently appearing in top 40%
    top_positions = [p for p in normalized_y_positions if p < 0.40]
    if len(top_positions) > len(sample_paths) * 0.3:  # If >30% of samples have top text
        top_margin = int(max(top_positions) * 100) + 40  # Add 40px buffer
    else:
        top_margin = 0

    # Find footer region: text consistently appearing in bottom 40%
    bottom_positions = [p for p in normalized_y_positions if p > 0.60]
    if len(bottom_positions) > len(sample_paths) * 0.3:  # If >30% of samples have bottom text
        bottom_margin = int((1 - min(bottom_positions)) * 100) + 40  # Add 40px buffer
    else:
        bottom_margin = 0

    # Find left margin: text consistently appearing in left 25%
    left_positions = [p for p in normalized_x_positions if p < 0.25]
    if len(left_positions) > len(sample_paths) * 0.2:  # If >20% of samples have left text
        left_margin = int(max(left_positions) * 100) + 30  # Add 30px buffer
    else:
        left_margin = 0

    # Find right margin: text consistently appearing in right 25%
    right_positions = [p for p in normalized_x_positions if p > 0.75]
    if len(right_positions) > len(sample_paths) * 0.2:  # If >20% of samples have right text
        right_margin = int((1 - min(right_positions)) * 100) + 30  # Add 30px buffer
    else:
        right_margin = 0

    print(f"Detected content region: top={top_margin}px, bottom={bottom_margin}px, left={left_margin}px, right={right_margin}px")
    return top_margin, bottom_margin, left_margin, right_margin


def page_has_header_footer(ocr_engine, image_path, top_margin, bottom_margin, left_margin=0, right_margin=0):
    """
    Quick check if a specific page has text in the header/footer/side regions.
    Returns: (has_header, has_footer, has_left, has_right)
    """
    if top_margin == 0 and bottom_margin == 0 and left_margin == 0 and right_margin == 0:
        return False, False, False, False

    with Image.open(image_path) as img:
        page_height = img.height
        page_width = img.width

    results = ocr_engine.predict(str(image_path))
    if not results:
        return False, False, False, False

    res = results[0]
    texts = list(res.get("rec_texts", []))
    boxes = list(res.get("rec_boxes", []))

    has_header = False
    has_footer = False
    has_left = False
    has_right = False

    for text, box in zip(texts, boxes):
        text = text.strip()
        if not text:
            continue

        y_center = (box[1] + box[3]) / 2
        x_center = (box[0] + box[2]) / 2

        # Check if text is in header region
        if top_margin > 0 and y_center < top_margin:
            has_header = True

        # Check if text is in footer region
        if bottom_margin > 0 and y_center > (page_height - bottom_margin):
            has_footer = True

        # Check if text is in left region
        if left_margin > 0 and x_center < left_margin:
            has_left = True

        # Check if text is in right region
        if right_margin > 0 and x_center > (page_width - right_margin):
            has_right = True

    return has_header, has_footer, has_left, has_right


def crop_image_to_content(image_path, top_margin, bottom_margin, left_margin=0, right_margin=0):
    """
    Crop image to exclude header/footer and side regions.
    Returns cropped image path or original if no margins.
    """
    if top_margin == 0 and bottom_margin == 0 and left_margin == 0 and right_margin == 0:
        return image_path

    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate crop region
        left = left_margin
        top = top_margin
        right = width - right_margin
        bottom = height - bottom_margin

        # Validate margins don't overlap
        if top >= bottom or left >= right:
            return image_path

        # Crop
        cropped = img.crop((left, top, right, bottom))

        # Save to temp file
        path = Path(image_path)
        temp_path = path.parent / (path.stem + '_cropped' + path.suffix)
        cropped.save(temp_path)
        return temp_path


def process_image(ocr_engine, path, template_path=DIVIDER_ICON_PATH,
                  top_margin=0, bottom_margin=0, left_margin=0, right_margin=0, check_header_footer=False):
    """
    Process image with optional header/footer and side margin cropping.
    Only crops if the page actually has header/footer/side text.
    """
    original_path = Path(path)
    cropped_path = None

    # Quick OCR check to determine if page is centered (title page) before cropping
    # Title pages shouldn't have header/footer cropping applied, but side margins should still be removed
    is_centered = False
    if top_margin > 0 or bottom_margin > 0:
        quick_results = ocr_engine.predict(str(path))
        if quick_results and quick_results[0]:
            quick_texts = list(quick_results[0].get("rec_texts", []))
            quick_boxes = list(quick_results[0].get("rec_boxes", []))
            quick_items = list(zip(quick_texts, quick_boxes))
            with Image.open(path) as img:
                page_width = img.width
            is_centered = is_centered_page(quick_items, page_width)

    # Calculate final margins: side margins always apply to all pages, header/footer only for non-centered pages
    # Navigation elements are consistent across all pages, so apply side margins without per-page checking
    actual_left = left_margin
    actual_right = right_margin

    if is_centered:
        actual_top = 0
        actual_bottom = 0
    else:
        if check_header_footer and (top_margin > 0 or bottom_margin > 0):
            has_header, has_footer, has_left, has_right = page_has_header_footer(ocr_engine, path, top_margin, bottom_margin, left_margin, right_margin)
            actual_top = top_margin if has_header else 0
            actual_bottom = bottom_margin if has_footer else 0
        else:
            actual_top = top_margin
            actual_bottom = bottom_margin

    # Apply cropping if any margins are specified
    if actual_top > 0 or actual_bottom > 0 or actual_left > 0 or actual_right > 0:
        cropped_path = crop_image_to_content(original_path, actual_top, actual_bottom, actual_left, actual_right)
        path = cropped_path

    results = ocr_engine.predict(str(path))
    if not results:
        # Clean up temp file if it was created
        if cropped_path and cropped_path.exists():
            try:
                cropped_path.unlink()
            except:
                pass
        return [], None, None, False

    res = results[0]
    texts = list(res.get("rec_texts", []))
    boxes = list(res.get("rec_boxes", []))

    items = sort_lines(texts, boxes)
    filtered = cleanup_ocr_items(items)
    
    # Filter out header/footer content based on text patterns
    filtered = filter_header_footer_content(filtered)

    if is_empty_page([t for t, _ in filtered]):
        # Clean up temp file if it was created
        if cropped_path and cropped_path.exists():
            try:
                cropped_path.unlink()
            except:
                pass
        return [], None, None, False

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

    # Calculate body-left and indent threshold for continuation detection
    body_left = None
    indent_threshold = None
    first_x = None
    first_text = None
    is_continuation = False
    
    if filtered:
        body_left, indent_threshold, _ = _body_left_and_threshold(filtered)

        # Get first line info
        first_x = filtered[0][1][0]
        first_text = filtered[0][0]

        # Check if first line is at body-left (continuation)
        # Also check if first line starts with whitespace, which indicates continuation
        starts_with_whitespace = first_text.startswith((' ', '　', '\t'))
        # Add a small tolerance (2 pixels) to handle OCR variations
        is_continuation = (first_x < indent_threshold + 2) or starts_with_whitespace

    # Clean up temp file if it was created
    if cropped_path and cropped_path.exists():
        try:
            cropped_path.unlink()
        except:
            pass

    return [text for text, _ in page_tuples], first_x, first_text, is_continuation


def main():
    parser = argparse.ArgumentParser(
        description="NovelScribe: batch OCR for Traditional Chinese novel screenshots"
    )
    parser.add_argument("-i", "--input-dir", default="input",
                        help="Directory containing the screenshot images")
    parser.add_argument("-o", "--output-file", default="output/output.txt",
                        help="Combined text output path")
    parser.add_argument(
        "-d", "--divider-icon",
        default=DIVIDER_ICON_PATH,
        help="Path to a small paragraph-divider icon template (PNG). If not found, no divider detection is performed.",
    )
    parser.add_argument(
        "-s", "--header-samples",
        type=int,
        default=10,
        help="Number of sample images to scan for auto-detecting header/footer regions. Set to 0 to disable.",
    )
    parser.add_argument(
        "-c", "--check-per-page",
        action="store_true",
        help="Check each page individually for header/footer before cropping. Slower but more accurate.",
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

    # Detect header/footer regions if enabled
    top_margin = 0
    bottom_margin = 0
    if args.header_samples > 0:
        print(f"Auto-detecting header/footer regions from {args.header_samples} sample image(s) ...")
        top_margin, bottom_margin, left_margin, right_margin = detect_content_region(ocr, image_paths, sample_count=args.header_samples)

    parts = []
    prev_centered = None
    for i, img_path in enumerate(image_paths, 1):
        name = Path(img_path).name
        print(f"[{i}/{len(image_paths)}] {name}", flush=True)
        page_lines, first_x, first_text, is_continuation = process_image(ocr, img_path, template_path=args.divider_icon,
                                   top_margin=top_margin, bottom_margin=bottom_margin,
                                   left_margin=left_margin, right_margin=right_margin,
                                   check_header_footer=args.check_per_page)
        if not page_lines:
            continue

        is_centered = not any(line.startswith("　　") for line in page_lines)

        # Check if this page starts with a continuation fragment
        # Don't apply continuation logic to centered pages (title pages)
        if (not is_centered and page_lines and is_continuation and 
            prev_centered is not None and not prev_centered):  # Previous was body text
            # Merge with previous page's last paragraph
            if parts:
                # Use the first paragraph from page_lines (not first_text which is just the first OCR line)
                first_paragraph = page_lines[0]
                # Remove leading whitespace and merge
                first_paragraph_clean = first_paragraph.lstrip(' 　\t')
                parts[-1] = parts[-1].rstrip() + first_paragraph_clean
                # Add the remaining paragraphs from this page with proper paragraph spacing
                for line in page_lines[1:]:
                    parts.append("\n\n" + line)
                # Skip normal processing for this page since we've merged everything
                prev_centered = is_centered
                continue

        if prev_centered is not None:
            if not prev_centered and is_centered:
                # After body text and before a centered/empty/graphical page: 5 empty lines.
                parts.append("\n\n\n\n\n\n")
            elif prev_centered or is_centered:
                # Centered pages get 3 empty lines of separation.
                parts.append("\n\n\n\n")
            else:
                # Body pages get 1 empty line of separation.
                parts.append("\n\n")

        if is_centered:
            parts.append("\n".join(page_lines))
        else:
            # page_lines are already properly formatted paragraphs from rebuild_page
            # Join them with double newlines for proper paragraph spacing
            parts.append("\n\n".join(page_lines))

        prev_centered = is_centered

    content = "".join(parts)
    content = ensure_em_dash_spacing(content)
    content = content.replace("\u2014", "\u2500")
    # Replace OCR misread '°' with '。' and remove surrounding spaces
    content = re.sub(r'\s*\u00b0\s*', '\u3002', content)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nDone. Combined output written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
