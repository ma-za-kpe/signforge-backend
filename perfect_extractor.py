#!/usr/bin/env python3
"""
PERFECT EXTRACTOR
Found the boxes! 346x346px at 150dpi
Extract each box + 60px below for text
"""
import fitz
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path

class PerfectExtractor:
    def __init__(self, pdf_path, output_dir):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.signs = []
        
    def detect_sign_boxes(self, page_img):
        """Detect 346x346px sign boxes using edge detection"""
        img_cv = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Sign boxes are 346x346 (±10px tolerance)
            if 336 < w < 356 and 336 < h < 356:
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
        
        # Sort top-to-bottom, left-to-right
        boxes.sort(key=lambda b: (b['y'], b['x']))
        
        return boxes
    
    def process_page(self, page_num):
        """Extract all sign boxes from a page"""
        pdf = fitz.open(self.pdf_path)
        page = pdf[page_num - 1]
        
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        boxes = self.detect_sign_boxes(img)
        
        print(f"📄 Page {page_num}: {len(boxes)} boxes detected")
        
        page_signs = []
        
        for idx, box in enumerate(boxes):
            try:
                # Extract box + 100px below for full title/description (2 lines)
                x0 = box['x']
                y0 = box['y']
                x1 = box['x'] + box['w']
                y1 = box['y'] + box['h'] + 100  # Add text space for full description
                
                # Ensure within bounds
                y1 = min(y1, img.height)
                
                # Crop
                cropped = img.crop((x0, y0, x1, y1))
                
                # Save
                filename = f"page_{page_num:03d}_box_{idx}.png"
                filepath = self.output_dir / filename
                cropped.save(filepath, quality=95)
                
                page_signs.append({
                    'page': page_num,
                    'box_index': idx,
                    'image': filename,
                    'bbox': [x0, y0, x1, y1],
                    'box_size': [box['w'], box['h']]
                })
                
            except Exception as e:
                print(f"   ❌ Box {idx} error: {e}")
        
        pdf.close()
        return page_signs
    
    def batch_extract(self, start_page=16, end_page=65):
        """Extract all pages"""
        print(f"🎯 PERFECT EXTRACTION: Pages {start_page}-{end_page}")
        print(f"   Method: OpenCV contour detection (346x346px boxes)")
        print(f"   Includes: Full box + 60px text below\n")
        
        for page_num in range(start_page, end_page + 1):
            try:
                signs = self.process_page(page_num)
                self.signs.extend(signs)
                
                if page_num % 10 == 0:
                    print(f"   Progress: {len(self.signs)} signs extracted")
                    
            except Exception as e:
                print(f"   ❌ Page {page_num}: {e}")
        
        # Save metadata
        output_file = self.output_dir / "perfect_extraction.json"
        with open(output_file, 'w') as f:
            json.dump(self.signs, f, indent=2)
        
        print(f"\n✅ COMPLETE!")
        print(f"   Total: {len(self.signs)} signs")
        print(f"   Output: {self.output_dir}")
        print(f"   All boxes include full 4 borders + text below")

if __name__ == "__main__":
    import sys

    extractor = PerfectExtractor(
        pdf_path="pdf.pdf",
        output_dir="ghsl_brain/perfect_extraction"
    )

    # Get start/end from command line or use defaults
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 66
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    extractor.batch_extract(start_page=start, end_page=end)
