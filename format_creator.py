"""
AGENT 3: FORMAT CREATOR
Generates multiple accessible formats from sign language content
- QR codes (offline access)
- Twi audio (local language, gTTS)
- PDF worksheets (printable)
- Video (future: MediaPipe)
- Haptic patterns (future: vibration JSON)
"""

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import qrcode
import requests
from gtts import gTTS
from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class FormatCreator:
    """Creates multiple accessibility formats for sign language content"""

    def __init__(self, brain_dir: Path, api_base_url: str = "http://localhost:9000"):
        self.brain_dir = Path(brain_dir)
        self.api_base_url = api_base_url
        self.output_dir = brain_dir / "generated_formats"
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for each format
        (self.output_dir / "qr_codes").mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)

    def create_qr_code(self, word: str, sign_image_url: str) -> Dict:
        """
        Create QR code that links to sign image
        For offline access: Students can scan and download
        """
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )

        # QR data includes word and image URL
        qr_data = {
            "word": word,
            "sign_image": sign_image_url,
            "source": "Ghana Sign Language Dictionary",
        }

        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save to file
        qr_filename = f"{word.replace(' ', '_').upper()}_QR.png"
        qr_path = self.output_dir / "qr_codes" / qr_filename
        img.save(qr_path)

        # Also return base64 for API
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "format": "qr_code",
            "word": word,
            "file_path": str(qr_path),
            "base64": img_base64,
            "data": qr_data,
        }

    def create_twi_audio(self, word: str, english_text: str) -> Dict:
        """
        Create Twi audio narration using gTTS
        Translates common words to Twi (Ghana's most common language)
        """
        # Simple English → Twi translation map (expand as needed)
        twi_translations = {
            "cow": "nantwi",
            "hello": "maakye",
            "thank you": "medaase",
            "good morning": "maakye",
            "good afternoon": "maaha",
            "good evening": "maadwo",
            "father": "papa",
            "mother": "maame",
            "school": "sukuu",
            "teacher": "okyerɛkyerɛfo",
            "student": "sukuuni",
            "water": "nsu",
            "food": "aduane",
            "love": "ɔdɔ",
            "one": "baako",
            "two": "mmienu",
            "three": "mmiɛnsa",
            "four": "ɛnan",
            "five": "enum",
        }

        # Get Twi translation or use English
        word_lower = word.lower()
        twi_word = twi_translations.get(word_lower, word)

        # Create audio with gTTS (using English for now, can switch to Akan/Twi if available)
        # Note: gTTS doesn't have native Twi, so we use transliterated text with English voice
        audio_text = f"{word}. In Twi: {twi_word}"

        try:
            tts = gTTS(text=audio_text, lang="en", slow=False)

            # Save to file
            audio_filename = f"{word.replace(' ', '_').upper()}_AUDIO.mp3"
            audio_path = self.output_dir / "audio" / audio_filename
            tts.save(str(audio_path))

            return {
                "format": "audio",
                "word": word,
                "file_path": str(audio_path),
                "language": "en-twi",
                "twi_translation": twi_word,
                "text": audio_text,
                "duration_estimate": len(audio_text) / 15,  # ~15 chars per second
            }
        except Exception as e:
            return {"format": "audio", "word": word, "error": str(e), "twi_translation": twi_word}

    def create_pdf_worksheet(
        self, words: List[str], lesson_title: str = "Sign Language Lesson"
    ) -> Dict:
        """
        Create printable PDF worksheet with signs
        For classrooms without digital access
        """
        pdf_filename = f"{lesson_title.replace(' ', '_')}_worksheet.pdf"
        pdf_path = self.output_dir / "pdfs" / pdf_filename

        # Create PDF
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(1 * inch, height - 1 * inch, lesson_title)

        # Subtitle
        c.setFont("Helvetica", 12)
        c.drawString(
            1 * inch, height - 1.3 * inch, "Ghana Sign Language Dictionary - Student Worksheet"
        )

        # Instructions
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(
            1 * inch, height - 1.6 * inch, "Practice these signs by looking at the images below:"
        )

        # Add signs (2 columns)
        y_position = height - 2.2 * inch
        x_positions = [1 * inch, 4.5 * inch]
        col = 0

        for idx, word in enumerate(words[:12]):  # Max 12 signs per page
            x = x_positions[col]

            # Word label
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x, y_position, f"{idx + 1}. {word.upper()}")

            # Try to add sign image if available
            sign_image_path = self.brain_dir / "sign_images" / f"{word.upper()}.png"
            if sign_image_path.exists():
                try:
                    img = ImageReader(str(sign_image_path))
                    c.drawImage(
                        img,
                        x,
                        y_position - 1.5 * inch,
                        width=2 * inch,
                        height=1.3 * inch,
                        preserveAspectRatio=True,
                    )
                except:
                    c.setFont("Helvetica", 10)
                    c.drawString(x, y_position - 0.5 * inch, "[Sign image placeholder]")
            else:
                c.setFont("Helvetica", 10)
                c.drawString(x, y_position - 0.5 * inch, "[Sign image not available]")

            # Practice boxes
            c.setFont("Helvetica", 9)
            c.drawString(x, y_position - 1.8 * inch, "Practice count: ___ times")

            # Move to next position
            col += 1
            if col >= 2:
                col = 0
                y_position -= 2.5 * inch

                # New page if needed
                if y_position < 2 * inch and idx < len(words) - 1:
                    c.showPage()
                    y_position = height - 2 * inch

        # Footer
        c.setFont("Helvetica", 8)
        c.drawString(1 * inch, 0.5 * inch, f"Generated by SignForge AI • Total signs: {len(words)}")
        c.drawString(
            1 * inch, 0.3 * inch, "Ghana Sign Language Dictionary 3rd Edition • UNICEF Ghana"
        )

        c.save()

        return {
            "format": "pdf",
            "lesson_title": lesson_title,
            "file_path": str(pdf_path),
            "total_signs": len(words),
            "pages": (len(words) // 12) + 1,
        }

    def create_haptic_pattern(self, word: str) -> Dict:
        """
        Create haptic (vibration) pattern JSON
        For deaf-blind students with vibration-enabled devices
        """
        # Map letters to vibration patterns
        # Short pulse = 100ms, Long pulse = 300ms, Gap = 100ms
        morse_like = {
            "A": [100, 100, 300],  # .-
            "B": [300, 100, 100, 100, 100, 100, 100],  # -...
            "C": [300, 100, 100, 100, 300],  # -.-.
            "D": [300, 100, 100, 100, 100],  # -..
            "E": [100],  # .
            "F": [100, 100, 100, 100, 300],  # ..-.
            "G": [300, 100, 300],  # --.
            "H": [100, 100, 100, 100, 100, 100, 100],  # ....
            "I": [100, 100, 100],  # ..
            "J": [100, 100, 300, 100, 300, 100, 300],  # .---
            "K": [300, 100, 100, 100, 300],  # -.-
            "L": [100, 100, 300, 100, 100],  # .-..
            "M": [300, 100, 300],  # --
            "N": [300, 100, 100],  # -.
            "O": [300, 100, 300, 100, 300],  # ---
            "P": [100, 100, 300, 100, 300],  # .--.
            "Q": [300, 100, 300, 100, 100, 100, 300],  # --.-
            "R": [100, 100, 300, 100, 100],  # .-.
            "S": [100, 100, 100, 100, 100],  # ...
            "T": [300],  # -
            "U": [100, 100, 100, 100, 300],  # ..-
            "V": [100, 100, 100, 100, 100, 100, 300],  # ...-
            "W": [100, 100, 300, 100, 300],  # .--
            "X": [300, 100, 100, 100, 100, 100, 300],  # -..-
            "Y": [300, 100, 100, 100, 300, 100, 300],  # -.--
            "Z": [300, 100, 300, 100, 100, 100, 100],  # --..
        }

        # Generate pattern for word
        pattern = []
        for char in word.upper():
            if char in morse_like:
                pattern.extend(morse_like[char])
                pattern.append(200)  # Gap between letters
            elif char == " ":
                pattern.append(500)  # Longer gap for spaces

        haptic_data = {
            "word": word,
            "pattern": pattern,
            "total_duration_ms": sum(pattern),
            "format": "vibration_pattern",
            "description": "Morse-code-like vibration pattern for haptic feedback",
        }

        # Save to JSON
        haptic_filename = f"{word.replace(' ', '_').upper()}_HAPTIC.json"
        haptic_path = self.output_dir / "videos" / haptic_filename  # Using videos dir for now

        with open(haptic_path, "w") as f:
            json.dump(haptic_data, f, indent=2)

        return {
            "format": "haptic",
            "word": word,
            "file_path": str(haptic_path),
            "pattern": pattern,
            "duration_ms": sum(pattern),
        }

    def create_all_formats(self, word: str, sign_image_url: str) -> Dict:
        """
        Create all formats for a single word
        Returns paths and metadata for all generated files
        """
        formats = {}

        # 1. QR Code
        try:
            formats["qr_code"] = self.create_qr_code(word, sign_image_url)
        except Exception as e:
            formats["qr_code"] = {"error": str(e)}

        # 2. Twi Audio
        try:
            formats["audio"] = self.create_twi_audio(word, word)
        except Exception as e:
            formats["audio"] = {"error": str(e)}

        # 3. Haptic Pattern
        try:
            formats["haptic"] = self.create_haptic_pattern(word)
        except Exception as e:
            formats["haptic"] = {"error": str(e)}

        return {
            "word": word,
            "sign_image_url": sign_image_url,
            "formats": formats,
            "total_formats": len([f for f in formats.values() if "error" not in f]),
        }

    def create_lesson_bundle(self, words: List[str], lesson_title: str) -> Dict:
        """
        Create complete lesson bundle with all formats for multiple words
        Perfect for teachers preparing a full lesson

        NOW INCLUDES: Phrase normalization for each word
        Fixes content generation issue - ensures "four" → 4.png, "and" → ALSO.png
        """
        from phrase_normalizer import get_phrase_normalizer

        bundle_results = {"lesson_title": lesson_title, "total_words": len(words), "formats": {}}

        # Get phrase normalizer
        normalizer = get_phrase_normalizer(self.brain_dir)

        # Get sign images for all words
        word_formats = []
        for word in words:
            # Get sign from API
            try:
                # NORMALIZE WORD FIRST (fixes content generation issue)
                normalized, matched_phrase = normalizer.normalize(word)

                # Use normalized word for image URL
                sign_image_url = f"{self.api_base_url}/sign_images/{normalized}.png"

                # Create formats with original word but normalized image
                formats = self.create_all_formats(word, sign_image_url)

                # Track normalization info for debugging
                formats["normalized_to"] = normalized
                if matched_phrase:
                    formats["matched_phrase"] = matched_phrase

                word_formats.append(formats)
            except Exception as e:
                word_formats.append({
                    "word": word,
                    "error": str(e),
                    "suggestion": "Check if sign exists in dictionary or try alternative word"
                })

        bundle_results["word_formats"] = word_formats

        # Create PDF worksheet for all words
        try:
            bundle_results["pdf_worksheet"] = self.create_pdf_worksheet(words, lesson_title)
        except Exception as e:
            bundle_results["pdf_worksheet"] = {"error": str(e)}

        return bundle_results


# Global instance
_format_creator: Optional[FormatCreator] = None


def get_format_creator(brain_dir: Path) -> FormatCreator:
    """Get or create format creator singleton"""
    global _format_creator
    if _format_creator is None:
        _format_creator = FormatCreator(brain_dir)
    return _format_creator
