#!/usr/bin/env python3
"""
EXTRACT PDF FRONT MATTER
Extract logos, messages, and content from first pages of PDF
"""
import fitz
from PIL import Image
import json
from pathlib import Path

class PDFContentExtractor:
    def __init__(self, pdf_path, output_dir):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.content = {}

    def extract_page_as_image(self, page_num, name):
        """Extract entire page as image"""
        pdf = fitz.open(self.pdf_path)
        page = pdf[page_num - 1]

        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        output_path = self.output_dir / f"{name}.png"
        img.save(output_path, quality=95)

        pdf.close()
        print(f"✅ Saved: {output_path.name}")
        return str(output_path.name)

    def extract_text_from_page(self, page_num):
        """Extract all text from a page"""
        pdf = fitz.open(self.pdf_path)
        page = pdf[page_num - 1]
        text = page.get_text()
        pdf.close()
        return text.strip()

    def extract_front_matter(self):
        """Extract content from first few pages"""
        print("🎯 EXTRACTING PDF FRONT MATTER\n")

        # Page 1: Cover/Title
        print("📄 Page 1: Cover")
        self.content['cover_image'] = self.extract_page_as_image(1, 'cover')
        self.content['cover_text'] = self.extract_text_from_page(1)

        # Page 2: Usually sponsors/credits
        print("📄 Page 2: Credits/Sponsors")
        self.content['page_2_image'] = self.extract_page_as_image(2, 'page_2_credits')
        self.content['page_2_text'] = self.extract_text_from_page(2)

        # Page 3: More info
        print("📄 Page 3: Additional Info")
        self.content['page_3_image'] = self.extract_page_as_image(3, 'page_3_info')
        self.content['page_3_text'] = self.extract_text_from_page(3)

        # Page 4-10: Look for forewords, acknowledgments
        print("\n📄 Pages 4-10: Messages and Forewords")
        messages = []
        for i in range(4, 11):
            text = self.extract_text_from_page(i)
            if text:
                # Check if this looks like a message/foreword
                if any(keyword in text.upper() for keyword in ['FOREWORD', 'MESSAGE', 'ACKNOWLEDGMENT', 'INTRODUCTION', 'PREFACE']):
                    img_file = self.extract_page_as_image(i, f'page_{i}_message')
                    messages.append({
                        'page': i,
                        'text': text,
                        'image': img_file
                    })
                    print(f"   Found message on page {i}")

        self.content['messages'] = messages

        # Save metadata
        metadata_file = self.output_dir / 'pdf_content.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.content, f, indent=2)

        print(f"\n✅ Saved metadata: {metadata_file}")
        print(f"\n📊 EXTRACTED CONTENT:")
        print(f"   Cover: {self.content['cover_image']}")
        print(f"   Credits page: {self.content['page_2_image']}")
        print(f"   Messages: {len(messages)} pages")

if __name__ == "__main__":
    extractor = PDFContentExtractor(
        pdf_path="pdf.pdf",
        output_dir="ghsl_brain/pdf_content"
    )

    extractor.extract_front_matter()
