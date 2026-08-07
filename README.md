# NovelScribe

Batch OCR tool for Traditional Chinese novel screenshots with automatic formatting and paragraph reconstruction.

## Features

- **Automatic Header/Footer Detection**: Samples pages to detect and remove header/footer regions before OCR
- **Cross-Page Continuation Detection**: Merges text fragments that span across page breaks
- **Paragraph Reconstruction**: Automatically detects paragraph boundaries and reconstructs paragraphs
- **Title/Body Text Recognition**: Distinguishes between centered titles and body text
- **Decorative Bar Removal**: Removes decorative bars that OCR misreads as digits or pipes
- **Em Dash Formatting**: Ensures proper spacing around em dashes and replaces them with standard dashes
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
python NovelScribe.py input_directory output_file
```

### Command-Line Options

```
positional arguments:
  input_dir             Directory containing the screenshot images (default: TestImage)
  output_file           Combined text output path (default: output\combined.txt)

options:
  --divider-icon PATH    Path to paragraph-divider icon template PNG
  --header-samples N     Number of sample images for header/footer detection (default: 10, 0 to disable)
  --check-per-page        Check each page individually for header/footer (slower but more accurate)
```

### Examples

```bash
# Basic usage with default settings
python NovelScribe.py TestImage output/combined.txt

# With custom header/footer detection samples
python NovelScribe.py TestImage output/combined.txt --header-samples 15

# Disable header/footer detection
python NovelScribe.py TestImage output/combined.txt --header-samples 0

# Check each page individually for header/footer
python NovelScribe.py TestImage output/combined.txt --check-per-page

# With custom paragraph divider icon
python NovelScribe.py TestImage output/combined.txt --divider-icon custom_icon.png
```

## How It Works

### Header/Footer Detection

The script samples pages to automatically detect header and footer regions:

1. Samples N images evenly distributed throughout the document
2. Runs OCR on sample pages to collect text positions
3. Analyzes text positions to find consistent header/footer regions
4. Crops images to exclude these regions before main OCR processing
5. Works universally with any novel format without manual configuration

### Cross-Page Continuation Detection

When a paragraph is split across pages:

1. Tracks whether the previous page was body text
2. Checks if the current page's first line is at body-left position (not indented)
3. If both conditions are met, merges the first line with the previous paragraph
4. Prevents fragments like "起" and "來" from appearing as separate lines

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

## Output Format

The output file contains:
- Body text with proper paragraph indentation ("　　")
- Centered titles and chapter numbers without indentation
- Proper spacing between different page types
- Em dashes (—) replaced with standard dashes (─)
- At least 3 empty lines before em-dash lines
- Paragraph divider markers ("---") where icons are detected

## File Naming

Input images should be named in the format: `Screenshot (N).png` where N is the page number.

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
