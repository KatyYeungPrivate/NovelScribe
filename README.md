# NovelScribe

Batch OCR tool for Traditional Chinese novel screenshots with automatic formatting and paragraph reconstruction.

## Features

- **Automatic Header/Footer Detection**: Samples pages to detect and remove header/footer regions before OCR
- **Header/Footer Content Filtering**: Automatically filters out navigation elements, page numbers, and titles from OCR output
- **Cross-Page Continuation Detection**: Merges text fragments that span across page breaks with OCR tolerance
- **Within-Page Continuation Detection**: Handles continuation lines at same indentation level
- **Paragraph Reconstruction**: Automatically detects paragraph boundaries and reconstructs paragraphs
- **Title/Body Text Recognition**: Distinguishes between centered titles and body text
- **Title Page Protection**: Skips header/footer cropping for centered title pages to preserve title text
- **Decorative Bar Removal**: Removes decorative bars that OCR misreads as digits or pipes
- **Em Dash Formatting**: Ensures proper spacing around em dashes and replaces them with standard dashes
- **OCR Error Correction**: Automatically corrects common OCR misreads (e.g., '°' → '。')
- **Paragraph Divider Detection**: Detects paragraph divider icons (if template provided)

## Requirements

- Python 3.11 or higher
- PaddlePaddle with GPU support (CUDA 12.6) or CPU mode
- Required packages:
  - paddlepaddle-gpu (for GPU) or paddlepaddle (for CPU)
  - paddleocr
  - pillow
  - opencv-python
  - numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KatyYeungPrivate/novel-scribe.git
cd novel-scribe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install paddlepaddle-gpu paddleocr pillow opencv-python numpy
```

## Usage

Basic usage:
```bash
python NovelScribe.py
```

### Command-Line Options

```
options:
  -i, --input-dir DIR      Directory containing the screenshot images (default: input)
  -o, --output-file PATH   Combined text output path (default: output/output.txt)
  -d, --divider-icon PATH  Path to paragraph-divider icon template PNG
  -s, --header-samples N   Number of sample images for header/footer detection (default: 10, 0 to disable)
  -c, --check-per-page     Check each page individually for header/footer (slower but more accurate)
```

### Examples

```bash
# Basic usage with default settings
python NovelScribe.py

# Specify custom input and output
python NovelScribe.py -i my_images -o my_output.txt

# With custom header/footer detection samples
python NovelScribe.py -i input -o output.txt --header-samples 15

# Disable header/footer detection
python NovelScribe.py -i input -o output.txt --header-samples 0

# Check each page individually for header/footer
python NovelScribe.py -i input -o output.txt --check-per-page

# With custom paragraph divider icon
python NovelScribe.py -i input -o output.txt --divider-icon custom_icon.png
```

## Page Capture Tool

A companion script `PageCapture.py` is included to help automate screenshot capture for web novels:

**Features:**
- Automatically presses Win+Screenshot followed by Right Arrow
- Configurable delay between actions
- Useful for capturing consecutive pages in web novels or similar content
- Saves screenshots to Windows Pictures\Screenshots folder

**Usage:**
```bash
python PageCapture.py
python PageCapture.py -n 10 -d 0.5  # 10 cycles with 0.5 second delay
```

**Options:**
- `-n, --count N`: Number of cycles (default: infinite, stop with Ctrl+C)
- `-d, --delay SECONDS`: Delay between actions in seconds (default: 1.0)

**Note:** Win+Screenshot saves PNG files to `Pictures\Screenshots` on Windows. You may need to rename/move these files to match the required naming format `(N).png` for NovelScribe.

## How It Works

### Header/Footer Detection

The script samples pages to automatically detect header and footer regions:

1. Samples N images evenly distributed throughout the document
2. Runs OCR on sample pages to collect text positions
3. Analyzes text positions to find consistent header/footer regions
4. Crops images to exclude these regions before main OCR processing
5. **Title Page Protection**: Skips cropping for centered pages (title pages) to preserve title text
6. Works universally with any novel format without manual configuration

### Header/Footer Content Filtering

After OCR processing, the script filters out header/footer content:

1. Removes navigation elements (e.g., "#449", "25%")
2. Removes page number patterns (e.g., "本章第11頁／共11頁")
3. Removes copyright symbols and metadata
4. Removes standalone book titles
5. Ensures only body text appears in the final output

### Cross-Page Continuation Detection

When a paragraph is split across pages:

1. Tracks whether the previous page was body text
2. Checks if the current page's first line is at body-left position (not indented)
3. Uses OCR tolerance (2 pixels) to handle detection variations
4. If both conditions are met, merges the first line with the previous paragraph
5. Prevents text fragments from appearing as separate lines

### Within-Page Continuation Detection

When continuation lines appear at the same indentation within a page:

1. Detects if the first line of a page is a continuation
2. Applies lenient paragraph detection for continuation pages
3. Requires significant indentation (+10 pixels) to start a new paragraph
4. Merges continuation lines that are at the same indentation level
5. Prevents false paragraph breaks in continuation text

### Paragraph Reconstruction

Within each page:

1. Calculates the body-left position (most common left edge)
2. Detects paragraph starts by checking indentation (body-left + ~2 chars)
3. Detects paragraph breaks by checking vertical gaps between lines
4. Adds proper indentation ("　　") to body text paragraphs
5. Keeps centered pages (titles, chapter numbers) without paragraph reconstruction

### Title/Body Text Recognition

- **Centered pages**: Text with mean x-position > 30% of page width (titles, chapter numbers)
- **Body text**: Text with consistent left edge and proper indentation
- Different spacing rules apply to each type

### OCR Error Correction

The script automatically corrects common OCR misreads:

- **Degree symbol correction**: Replaces '°' (degree symbol) with '。' (Chinese period)
- **Space removal**: Removes surrounding spaces when applying corrections
- Prevents common OCR errors from appearing in the final output

## Output Format

The output file contains:
- Body text with proper paragraph indentation ("　　")
- Centered titles and chapter numbers without indentation
- Proper spacing between different page types
- Em dashes (—) replaced with standard dashes (─)
- At least 3 empty lines before em-dash lines
- Paragraph divider markers ("---") where icons are detected
- OCR corrections applied (e.g., '°' → '。')

## File Naming

Input images must be named with a number in parentheses for proper sorting: `any_name (N).png` where N is the page number. The script extracts the number from parentheses using regex pattern `\((\d+)\)` to sort files numerically. Examples: `page (1).png`, `image (5).png`, `Screenshot (10).png`

## Troubleshooting

### GPU/CUDA Issues

If you encounter CUDA compatibility warnings, the script will automatically fall back to CPU mode. For better performance, ensure:
- CUDA version matches your PaddlePaddle installation
- GPU drivers are up to date

### Continuation Fragments

If you still see continuation fragments in the output:
- Increase `--header-samples` for more accurate header/footer detection
- Use `--check-per-page` to handle mixed page layouts
- Manually edit the output file if needed

### Empty Pages

Pages with no CJK text (illustrations, blank pages) are automatically skipped.

## License

This project is for personal use for OCR processing of Traditional Chinese novels.
