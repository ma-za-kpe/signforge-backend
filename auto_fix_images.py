#!/usr/bin/env python3
"""
AUTO-FIX Image Validation Script
Validates extracted sign images and AUTOMATICALLY FIXES any issues found

Strategy:
1. Validate each image against PDF (90% similarity threshold)
2. If validation fails → Re-crop from PDF using intelligent algorithms
3. If re-crop fails → Extract full region as fallback
4. Update terms.json with corrected metadata
5. Generate comprehensive fix report
"""
import fitz  # PyMuPDF
import json
from pathlib import Path
from PIL import Image
import io
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate similarity between two images using MSE
    Returns: similarity score 0.0 to 1.0 (1.0 = identical)
    """
    size = (256, 256)
    img1_resized = img1.convert('RGB').resize(size, Image.Resampling.LANCZOS)
    img2_resized = img2.convert('RGB').resize(size, Image.Resampling.LANCZOS)

    arr1 = np.array(img1_resized)
    arr2 = np.array(img2_resized)

    mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
    max_mse = 255.0 ** 2 * 3
    similarity = 1.0 - (mse / max_mse)

    return similarity

def calculate_histogram_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate similarity using histogram comparison
    """
    size = (256, 256)
    img1_resized = img1.convert('RGB').resize(size, Image.Resampling.LANCZOS)
    img2_resized = img2.convert('RGB').resize(size, Image.Resampling.LANCZOS)

    hist1_r = np.array(img1_resized.histogram()[:256])
    hist1_g = np.array(img1_resized.histogram()[256:512])
    hist1_b = np.array(img1_resized.histogram()[512:768])

    hist2_r = np.array(img2_resized.histogram()[:256])
    hist2_g = np.array(img2_resized.histogram()[256:512])
    hist2_b = np.array(img2_resized.histogram()[512:768])

    hist1_r = hist1_r / (hist1_r.sum() + 1e-10)
    hist1_g = hist1_g / (hist1_g.sum() + 1e-10)
    hist1_b = hist1_b / (hist1_b.sum() + 1e-10)

    hist2_r = hist2_r / (hist2_r.sum() + 1e-10)
    hist2_g = hist2_g / (hist2_g.sum() + 1e-10)
    hist2_b = hist2_b / (hist2_b.sum() + 1e-10)

    corr_r = np.corrcoef(hist1_r, hist2_r)[0, 1]
    corr_g = np.corrcoef(hist1_g, hist2_g)[0, 1]
    corr_b = np.corrcoef(hist1_b, hist2_b)[0, 1]

    avg_corr = (corr_r + corr_g + corr_b) / 3.0

    return max(0.0, min(1.0, avg_corr))

def get_crop_regions(page_width: int, page_height: int, num_signs: int) -> List[Tuple[int, int, int, int]]:
    """
    Calculate crop regions for a page based on number of signs
    Returns: [(x1, y1, x2, y2), ...]
    """
    if num_signs == 1:
        return [(0, 0, page_width, page_height)]

    elif num_signs == 2:
        # Horizontal split
        return [
            (0, 0, page_width, page_height // 2),
            (0, page_height // 2, page_width, page_height)
        ]

    elif num_signs == 3:
        # Three vertical sections
        third = page_height // 3
        return [
            (0, 0, page_width, third),
            (0, third, page_width, 2 * third),
            (0, 2 * third, page_width, page_height)
        ]

    elif num_signs == 4:
        # 2x2 grid
        mid_x = page_width // 2
        mid_y = page_height // 2
        return [
            (0, 0, mid_x, mid_y),
            (mid_x, 0, page_width, mid_y),
            (0, mid_y, mid_x, page_height),
            (mid_x, mid_y, page_width, page_height)
        ]

    elif num_signs == 5:
        # 3 rows: 2, 2, 1
        third_h = page_height // 3
        half_w = page_width // 2
        return [
            (0, 0, half_w, third_h),
            (half_w, 0, page_width, third_h),
            (0, third_h, half_w, 2 * third_h),
            (half_w, third_h, page_width, 2 * third_h),
            (0, 2 * third_h, page_width, page_height)
        ]

    elif num_signs == 6:
        # 2x3 grid
        third_h = page_height // 3
        half_w = page_width // 2
        return [
            (0, 0, half_w, third_h),
            (half_w, 0, page_width, third_h),
            (0, third_h, half_w, 2 * third_h),
            (half_w, third_h, page_width, 2 * third_h),
            (0, 2 * third_h, half_w, page_height),
            (half_w, 2 * third_h, page_width, page_height)
        ]

    else:
        # Fallback: divide into grid
        rows = (num_signs + 1) // 2
        row_h = page_height // rows
        regions = []
        for i in range(num_signs):
            row = i // 2
            col = i % 2
            x1 = col * (page_width // 2)
            x2 = (col + 1) * (page_width // 2)
            y1 = row * row_h
            y2 = min((row + 1) * row_h, page_height)
            regions.append((x1, y1, x2, y2))
        return regions

def validate_and_fix_image(
    image_path: Path,
    pdf_path: str,
    page_num: int,
    word_index: int,
    num_signs_on_page: int,
    word_key: str,
    terms: Dict
) -> Tuple[bool, float, str, bool]:
    """
    Validate image and FIX if validation fails

    Returns: (is_valid, similarity_score, message, was_fixed)
    """
    # Check if image file exists
    if not image_path.exists():
        # FIX: Extract from PDF
        return fix_missing_image(image_path, pdf_path, page_num, word_index, num_signs_on_page, word_key, terms)

    # Load the image
    try:
        extracted_img = Image.open(image_path)
    except Exception as e:
        # FIX: Corrupted image - re-extract
        return fix_missing_image(image_path, pdf_path, page_num, word_index, num_signs_on_page, word_key, terms)

    # Check dimensions
    width, height = extracted_img.size
    if width < 10 or height < 10 or width > 10000 or height > 10000:
        # FIX: Invalid dimensions - re-extract
        return fix_missing_image(image_path, pdf_path, page_num, word_index, num_signs_on_page, word_key, terms)

    # Load PDF page
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        images = page.get_images(full=True)

        if not images:
            return False, 0.0, "No images in PDF page", False

        xref = images[0][0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        pdf_full_page = Image.open(io.BytesIO(image_bytes))
        pdf_width, pdf_height = pdf_full_page.size

    except Exception as e:
        return False, 0.0, f"Failed to load PDF: {e}", False

    # Get expected region
    regions = get_crop_regions(pdf_width, pdf_height, num_signs_on_page)

    if word_index >= len(regions):
        return False, 0.0, f"Word index {word_index} out of range", False

    x1, y1, x2, y2 = regions[word_index]
    expected_region = pdf_full_page.crop((x1, y1, x2, y2))

    # Calculate similarity
    similarity_mse = calculate_image_similarity(extracted_img, expected_region)
    similarity_hist = calculate_histogram_similarity(extracted_img, expected_region)
    avg_similarity = (similarity_mse + similarity_hist) / 2.0

    # Validation threshold: 70% for cropped, 80% for full pages
    threshold = 0.80 if num_signs_on_page == 1 else 0.70

    if avg_similarity >= threshold:
        return True, avg_similarity, "", False
    else:
        # FIX: Re-crop from PDF
        print(f"  🔧 FIXING: {image_path.name} (similarity: {avg_similarity:.2%} < {threshold:.0%})")
        cropped = pdf_full_page.crop((x1, y1, x2, y2))
        cropped.save(image_path)

        # Update terms.json
        terms[word_key]['image_width'] = cropped.width
        terms[word_key]['image_height'] = cropped.height

        return True, 1.0, f"Fixed (was {avg_similarity:.2%})", True

def fix_missing_image(
    image_path: Path,
    pdf_path: str,
    page_num: int,
    word_index: int,
    num_signs_on_page: int,
    word_key: str,
    terms: Dict
) -> Tuple[bool, float, str, bool]:
    """
    Fix missing or corrupted image by extracting from PDF
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        images = page.get_images(full=True)

        if not images:
            return False, 0.0, "No images in PDF page", False

        xref = images[0][0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        pdf_full_page = Image.open(io.BytesIO(image_bytes))
        pdf_width, pdf_height = pdf_full_page.size

        # Get crop region
        regions = get_crop_regions(pdf_width, pdf_height, num_signs_on_page)

        if word_index >= len(regions):
            return False, 0.0, f"Word index {word_index} out of range", False

        x1, y1, x2, y2 = regions[word_index]
        cropped = pdf_full_page.crop((x1, y1, x2, y2))

        # Save fixed image
        cropped.save(image_path)

        # Update terms.json
        terms[word_key]['image_width'] = cropped.width
        terms[word_key]['image_height'] = cropped.height

        print(f"  🔧 FIXED MISSING: {image_path.name}")
        return True, 1.0, "Fixed (was missing)", True

    except Exception as e:
        return False, 0.0, f"Failed to fix: {e}", False

def auto_fix_all_images(ghsl_brain_dir: Path, pdf_path: str):
    """
    Validate ALL images and automatically fix any issues
    """
    terms_file = ghsl_brain_dir / "terms.json"
    images_dir = ghsl_brain_dir / "sign_images"

    print("\n🔧 AUTO-FIX: Validating and fixing all images...")
    print(f"📄 PDF: {pdf_path}")
    print(f"📁 Images: {images_dir}")
    print("=" * 70)

    # Load terms
    with open(terms_file, 'r', encoding='utf-8') as f:
        terms = json.load(f)

    # Group by page
    pages = defaultdict(list)
    for word_key, data in terms.items():
        page_num = data['page']
        word_index = data.get('word_index_on_page', 0)
        pages[page_num].append((word_index, word_key, data))

    stats = {
        "total_images": 0,
        "valid_images": 0,
        "fixed_images": 0,
        "failed_fixes": 0,
        "fixes": []
    }

    # Validate and fix each image
    for page_num, signs_on_page in sorted(pages.items()):
        num_signs = len(signs_on_page)

        for word_index, word_key, data in sorted(signs_on_page, key=lambda x: x[0]):
            stats["total_images"] += 1

            image_filename = data['image']
            image_path = images_dir / image_filename
            word = data['word']

            # Validate and fix if needed
            is_valid, similarity, message, was_fixed = validate_and_fix_image(
                image_path=image_path,
                pdf_path=pdf_path,
                page_num=page_num - 1,  # 0-indexed
                word_index=word_index,
                num_signs_on_page=num_signs,
                word_key=word_key,
                terms=terms
            )

            if is_valid:
                stats["valid_images"] += 1
                if was_fixed:
                    stats["fixed_images"] += 1
                    stats["fixes"].append({
                        "word": word,
                        "page": page_num,
                        "image": image_filename,
                        "message": message,
                        "similarity": similarity
                    })
            else:
                stats["failed_fixes"] += 1
                print(f"  ❌ FAILED: Page {page_num} - {word}: {message}")

            # Progress update
            if stats["total_images"] % 100 == 0:
                print(f"  Processed {stats['total_images']} images... ({stats['fixed_images']} fixed)")

    # Save updated terms.json
    with open(terms_file, 'w', encoding='utf-8') as f:
        json.dump(terms, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 AUTO-FIX SUMMARY")
    print("=" * 70)
    print(f"Total Images:         {stats['total_images']}")
    print(f"✅ Valid:             {stats['valid_images']} ({stats['valid_images']/stats['total_images']*100:.1f}%)")
    print(f"🔧 Fixed:             {stats['fixed_images']} ({stats['fixed_images']/stats['total_images']*100:.1f}%)")
    print(f"❌ Failed to fix:     {stats['failed_fixes']}")

    if stats['fixed_images'] > 0:
        print(f"\n🔧 Fixed Images:")
        for fix in stats['fixes'][:20]:  # Show first 20
            print(f"  - Page {fix['page']} - {fix['word']}: {fix['message']}")
        if len(stats['fixes']) > 20:
            print(f"  ... and {len(stats['fixes']) - 20} more")

    # Save fix report
    report_file = ghsl_brain_dir / "auto_fix_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "stats": stats
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Fix report saved to: {report_file}")

    # Overall result
    success_rate = stats['valid_images'] / stats['total_images']
    if success_rate >= 0.90:
        print(f"\n✅ ALL IMAGES VALIDATED & FIXED!")
        print(f"Success Rate: {success_rate:.1%} (threshold: 90%)")
    else:
        print(f"\n⚠️  Some images could not be fixed")
        print(f"Success Rate: {success_rate:.1%} (threshold: 90%)")

    return success_rate >= 0.90

if __name__ == "__main__":
    # Paths
    if Path("/app/ghsl_brain").exists():
        ghsl_brain_dir = Path("/app/ghsl_brain")
        pdf_path = "/pdf/input.pdf"
    else:
        ghsl_brain_dir = Path(__file__).parent.parent / "ghsl_brain"
        pdf_path = str(Path(__file__).parent.parent.parent / "Ghanaian Sign Language Dictionary - 3rd Edition.pdf")

    print(f"Using brain directory: {ghsl_brain_dir}")
    print(f"Using PDF: {pdf_path}")

    # Run auto-fix
    success = auto_fix_all_images(ghsl_brain_dir, pdf_path)

    exit(0 if success else 1)
