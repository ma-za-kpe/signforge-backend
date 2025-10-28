"""
AGENT 4: RURAL DELIVERY SERVICE
Delivers sign language content to rural areas via SMS, USSD (*123#), and WhatsApp

Features:
- USSD menu system (*123# for feature phones)
- SMS delivery with short links
- WhatsApp bot integration
- Offline pack generation and distribution
- Village chief notifications
- Data usage optimization
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class DeliveryChannel(Enum):
    """Supported delivery channels"""
    SMS = "sms"
    USSD = "ussd"
    WHATSAPP = "whatsapp"


class RuralDeliveryService:
    """
    Manages delivery of sign language content to rural areas
    Optimized for:
    - Feature phones (USSD)
    - Limited data plans (compressed images, short links)
    - Offline access (downloadable packs)
    - Low literacy (voice + images)
    """

    def __init__(self, brain_dir: Path, config: Optional[Dict] = None):
        self.brain_dir = Path(brain_dir)
        self.config = config or {}

        # Initialize delivery providers
        self.africastalking_api_key = self.config.get('AFRICASTALKING_API_KEY', os.getenv('AFRICASTALKING_API_KEY'))
        self.africastalking_username = self.config.get('AFRICASTALKING_USERNAME', os.getenv('AFRICASTALKING_USERNAME', 'sandbox'))

        self.twilio_account_sid = self.config.get('TWILIO_ACCOUNT_SID', os.getenv('TWILIO_ACCOUNT_SID'))
        self.twilio_auth_token = self.config.get('TWILIO_AUTH_TOKEN', os.getenv('TWILIO_AUTH_TOKEN'))
        self.twilio_whatsapp_from = self.config.get('TWILIO_WHATSAPP_FROM', os.getenv('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886'))

        # Base URL for sign resources
        self.base_url = self.config.get('BASE_URL', os.getenv('BASE_URL', 'http://localhost:9000'))

        # Short link service (optional, can use TinyURL API or custom)
        self.use_short_links = self.config.get('USE_SHORT_LINKS', True)

        # Delivery logs
        self.delivery_log_file = brain_dir / "delivery_log.json"
        if not self.delivery_log_file.exists():
            self._init_delivery_log()

    def _init_delivery_log(self):
        """Initialize delivery log file"""
        default_log = {
            "total_deliveries": 0,
            "by_channel": {
                "sms": 0,
                "ussd": 0,
                "whatsapp": 0
            },
            "by_region": {},
            "deliveries": []
        }
        with open(self.delivery_log_file, 'w') as f:
            json.dump(default_log, f, indent=2)

    def _log_delivery(self, channel: DeliveryChannel, phone: str, word: str, success: bool, details: Optional[Dict] = None):
        """Log delivery event"""
        with open(self.delivery_log_file, 'r') as f:
            log = json.load(f)

        log["total_deliveries"] += 1
        log["by_channel"][channel.value] += 1

        log["deliveries"].append({
            "timestamp": datetime.now().isoformat(),
            "channel": channel.value,
            "phone": phone,
            "word": word,
            "success": success,
            "details": details or {}
        })

        # Keep only last 1000 deliveries
        log["deliveries"] = log["deliveries"][-1000:]

        with open(self.delivery_log_file, 'w') as f:
            json.dump(log, f, indent=2)

    def _shorten_url(self, url: str) -> str:
        """
        Shorten URL using TinyURL API (free, no API key required)
        Falls back to original URL if shortening fails
        """
        if not self.use_short_links:
            return url

        try:
            response = requests.get(f"http://tinyurl.com/api-create.php?url={url}", timeout=5)
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            pass

        return url

    # ============================================
    # SMS DELIVERY
    # ============================================

    def send_sms(self, phone: str, word: str) -> Dict:
        """
        Send SMS with sign link to phone number

        Example SMS:
        "✅ SIGN: COW
        📷 Image: https://short.link/cow.png
        🔊 Audio: https://short.link/cow.mp3
        Reply HELP for more options
        - SignForge"
        """
        # Get sign image URL
        sign_image_url = f"{self.base_url}/sign_images/{word.upper()}.png"

        # Format creator paths
        from format_creator import get_format_creator
        creator = get_format_creator(self.brain_dir)

        # Generate short links
        image_link = self._shorten_url(sign_image_url)

        # Check if audio exists
        audio_file = creator.output_dir / "audio" / f"{word.upper().replace(' ', '_')}_AUDIO.mp3"
        audio_link = None
        if audio_file.exists():
            audio_link = self._shorten_url(f"{self.base_url}/generated_formats/audio/{audio_file.name}")

        # Compose SMS message (160 chars max for single SMS)
        message = f"✅ SIGN: {word.upper()}\n"
        message += f"📷 {image_link}\n"
        if audio_link:
            message += f"🔊 {audio_link}\n"
        message += "Reply HELP for more\n- SignForge"

        # Send via Africa's Talking (or fallback to simulation)
        if self.africastalking_api_key:
            try:
                result = self._send_africastalking_sms(phone, message)
                self._log_delivery(DeliveryChannel.SMS, phone, word, True, result)
                return {
                    "success": True,
                    "channel": "sms",
                    "phone": phone,
                    "word": word,
                    "message": message,
                    "delivery_result": result
                }
            except Exception as e:
                self._log_delivery(DeliveryChannel.SMS, phone, word, False, {"error": str(e)})
                return {
                    "success": False,
                    "channel": "sms",
                    "error": str(e)
                }
        else:
            # Simulation mode (no API key)
            self._log_delivery(DeliveryChannel.SMS, phone, word, True, {"mode": "simulation"})
            return {
                "success": True,
                "channel": "sms",
                "phone": phone,
                "word": word,
                "message": message,
                "mode": "simulation",
                "note": "Set AFRICASTALKING_API_KEY to enable real SMS delivery"
            }

    def _send_africastalking_sms(self, phone: str, message: str) -> Dict:
        """Send SMS via Africa's Talking API"""
        url = "https://api.africastalking.com/version1/messaging"

        headers = {
            "apiKey": self.africastalking_api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "username": self.africastalking_username,
            "to": phone,
            "message": message
        }

        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()

        return response.json()

    # ============================================
    # USSD MENU SYSTEM
    # ============================================

    def handle_ussd_request(self, session_id: str, service_code: str, phone_number: str, text: str) -> Tuple[str, bool]:
        """
        Handle USSD menu navigation

        USSD Flow:
        *123#
        > Welcome to SignForge
        > 1. Search for sign
        > 2. Download lesson
        > 3. Get help

        Returns: (response_text, end_session)
        """
        # Parse user input
        user_input = text.strip()

        # Main menu
        if not user_input:
            response = "CON Welcome to SignForge 🤟\n"
            response += "Ghana Sign Language\n\n"
            response += "1. Search for sign\n"
            response += "2. Download lesson\n"
            response += "3. Popular signs\n"
            response += "4. Get help"
            return (response, False)

        # Split input to track navigation
        parts = user_input.split('*')

        # Option 1: Search for sign
        if parts[0] == '1':
            if len(parts) == 1:
                response = "CON Enter word to search:\n"
                response += "(e.g., cow, hello, thank you)"
                return (response, False)
            else:
                # User entered a word
                word = ' '.join(parts[1:]).strip()

                # Search for sign
                try:
                    from hybrid_search_service import get_hybrid_search_service
                    search = get_hybrid_search_service(self.brain_dir)
                    results = search.search(word, top_k=1)

                    if results:
                        sign = results[0]
                        response = f"END ✅ SIGN FOUND: {sign['word']}\n\n"
                        response += f"Image will be sent to {phone_number} via SMS\n\n"
                        response += "Thank you for using SignForge!"

                        # Send SMS with sign link
                        self.send_sms(phone_number, sign['word'])

                        return (response, True)
                    else:
                        response = f"END ❌ No sign found for '{word}'\n\n"
                        response += "Try another word or contact support."
                        return (response, True)
                except Exception as e:
                    response = f"END ⚠️ Error: {str(e)}\n\n"
                    response += "Please try again later."
                    return (response, True)

        # Option 2: Download lesson
        elif parts[0] == '2':
            if len(parts) == 1:
                response = "CON Available lessons:\n\n"
                response += "1. Farm Animals\n"
                response += "2. Greetings\n"
                response += "3. Numbers\n"
                response += "4. Colors\n"
                response += "0. Back to main menu"
                return (response, False)
            else:
                lesson_map = {
                    '1': 'Farm Animals',
                    '2': 'Greetings',
                    '3': 'Numbers',
                    '4': 'Colors'
                }

                lesson_choice = parts[1]

                if lesson_choice == '0':
                    # Restart session by returning main menu
                    return self.handle_ussd_request(session_id, service_code, phone_number, '')

                if lesson_choice in lesson_map:
                    lesson_name = lesson_map[lesson_choice]
                    response = f"END 📚 {lesson_name} lesson\n\n"
                    response += f"Download link will be sent to {phone_number} via SMS\n\n"
                    response += "Data charges may apply\n"
                    response += "Thank you for using SignForge!"

                    # TODO: Send SMS with lesson download link

                    return (response, True)
                else:
                    response = "END Invalid option. Please try again."
                    return (response, True)

        # Option 3: Popular signs
        elif parts[0] == '3':
            response = "END 🔥 Most searched signs:\n\n"
            response += "1. Hello\n"
            response += "2. Thank you\n"
            response += "3. Good morning\n"
            response += "4. Love\n"
            response += "5. Water\n\n"
            response += "Dial *123#1 to search"
            return (response, True)

        # Option 4: Help
        elif parts[0] == '4':
            response = "END ℹ️ SignForge Help\n\n"
            response += "📞 Call: 0800-123-4567\n"
            response += "📧 Email: help@signforge.gh\n"
            response += "🌐 Web: signforge.gh\n\n"
            response += "Powered by UNICEF Ghana"
            return (response, True)

        # Invalid input
        else:
            response = "END Invalid option. Please dial *123# to start again."
            return (response, True)

    # ============================================
    # WHATSAPP BOT
    # ============================================

    def send_whatsapp(self, phone: str, word: str) -> Dict:
        """
        Send WhatsApp message with sign content

        Example WhatsApp message:
        "🐄 COW - Ghana Sign Language

        [Image of COW sign]

        🔊 Audio pronunciation (Twi: nantwi)
        [Audio file attached]

        📚 Related signs: FARM, MILK, GRASS
        📥 Download lesson: Reply 'LESSON farm-animals'

        Powered by SignForge • UNICEF Ghana"
        """
        # Get sign image
        sign_image_path = self.brain_dir / "sign_images" / f"{word.upper()}.png"

        if not sign_image_path.exists():
            return {
                "success": False,
                "error": f"Sign image not found for '{word}'"
            }

        # Format message
        message = f"🤟 {word.upper()} - Ghana Sign Language\n\n"
        message += f"📷 Sign image attached\n\n"

        # Get Twi translation if available
        from format_creator import FormatCreator
        creator = FormatCreator(self.brain_dir)
        twi_translations = {
            "cow": "nantwi",
            "hello": "maakye",
            "thank you": "medaase",
        }

        twi_word = twi_translations.get(word.lower())
        if twi_word:
            message += f"🇬🇭 Twi: {twi_word}\n\n"

        message += f"📚 Related signs: Reply 'MORE {word.upper()}' for similar signs\n"
        message += f"📥 Get lesson: Reply 'LESSON' for full lessons\n\n"
        message += "Powered by SignForge • UNICEF Ghana"

        # Send via Twilio WhatsApp (or simulation)
        if self.twilio_account_sid and self.twilio_auth_token:
            try:
                result = self._send_twilio_whatsapp(phone, message, str(sign_image_path))
                self._log_delivery(DeliveryChannel.WHATSAPP, phone, word, True, result)
                return {
                    "success": True,
                    "channel": "whatsapp",
                    "phone": phone,
                    "word": word,
                    "message": message,
                    "delivery_result": result
                }
            except Exception as e:
                self._log_delivery(DeliveryChannel.WHATSAPP, phone, word, False, {"error": str(e)})
                return {
                    "success": False,
                    "channel": "whatsapp",
                    "error": str(e)
                }
        else:
            # Simulation mode
            self._log_delivery(DeliveryChannel.WHATSAPP, phone, word, True, {"mode": "simulation"})
            return {
                "success": True,
                "channel": "whatsapp",
                "phone": phone,
                "word": word,
                "message": message,
                "mode": "simulation",
                "note": "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to enable real WhatsApp delivery"
            }

    def _send_twilio_whatsapp(self, phone: str, message: str, image_path: Optional[str] = None) -> Dict:
        """Send WhatsApp message via Twilio API"""
        from twilio.rest import Client

        client = Client(self.twilio_account_sid, self.twilio_auth_token)

        # Format phone number for WhatsApp
        whatsapp_to = f"whatsapp:{phone}"

        # Send message with optional image
        if image_path:
            # Upload image and get public URL (simplified - in production, use cloud storage)
            media_url = [f"{self.base_url}/sign_images/{Path(image_path).name}"]

            msg = client.messages.create(
                from_=self.twilio_whatsapp_from,
                to=whatsapp_to,
                body=message,
                media_url=media_url
            )
        else:
            msg = client.messages.create(
                from_=self.twilio_whatsapp_from,
                to=whatsapp_to,
                body=message
            )

        return {
            "message_sid": msg.sid,
            "status": msg.status,
            "to": phone
        }

    def handle_whatsapp_webhook(self, message_body: str, from_phone: str) -> Dict:
        """
        Handle incoming WhatsApp messages

        Supported commands:
        - "cow" → Send COW sign
        - "LESSON" → List available lessons
        - "LESSON farm-animals" → Send farm animals lesson
        - "MORE cow" → Related signs to COW
        - "HELP" → Help message
        """
        command = message_body.strip().lower()

        # Help command
        if command == 'help':
            response = "🤟 SignForge WhatsApp Bot\n\n"
            response += "Commands:\n"
            response += "• Send any word (e.g., 'cow') to get its sign\n"
            response += "• LESSON - View available lessons\n"
            response += "• MORE <word> - Get related signs\n"
            response += "• HELP - This message\n\n"
            response += "Powered by UNICEF Ghana"

            return {
                "response": response,
                "action": "help"
            }

        # Lesson command
        if command.startswith('lesson'):
            parts = command.split()
            if len(parts) == 1:
                response = "📚 Available Lessons:\n\n"
                response += "1. farm-animals\n"
                response += "2. greetings\n"
                response += "3. numbers\n"
                response += "4. colors\n\n"
                response += "Reply: LESSON farm-animals"

                return {
                    "response": response,
                    "action": "list_lessons"
                }
            else:
                lesson_name = parts[1]
                response = f"📥 Sending '{lesson_name}' lesson...\n\n"
                response += "This may take a moment. Large file."

                # TODO: Generate and send lesson pack

                return {
                    "response": response,
                    "action": "send_lesson",
                    "lesson": lesson_name
                }

        # More/related signs command
        if command.startswith('more'):
            parts = command.split()
            if len(parts) > 1:
                word = parts[1]
                response = f"📚 Signs related to '{word.upper()}':\n\n"
                response += "• FARM\n"
                response += "• MILK\n"
                response += "• GRASS\n\n"
                response += "Reply with any word to see its sign"

                return {
                    "response": response,
                    "action": "related_signs",
                    "word": word
                }

        # Default: treat as sign search
        word = command
        try:
            # Send sign via WhatsApp
            result = self.send_whatsapp(from_phone, word)

            return {
                "response": f"Sending sign for '{word.upper()}'...",
                "action": "send_sign",
                "word": word,
                "result": result
            }
        except Exception as e:
            return {
                "response": f"❌ Could not find sign for '{word}'. Try another word or type HELP.",
                "action": "error",
                "error": str(e)
            }

    # ============================================
    # OFFLINE PACK GENERATION
    # ============================================

    def create_offline_pack(self, lesson_title: str, words: List[str]) -> Dict:
        """
        Create downloadable offline pack for rural areas

        Pack includes:
        - All sign images (compressed)
        - Audio files (compressed)
        - PDF worksheet
        - HTML viewer (works offline)
        - Total size optimized for 2G/3G networks
        """
        import zipfile
        from PIL import Image

        pack_dir = self.brain_dir / "offline_packs"
        pack_dir.mkdir(exist_ok=True)

        pack_name = f"{lesson_title.replace(' ', '_').lower()}_offline.zip"
        pack_path = pack_dir / pack_name

        # Create ZIP file
        with zipfile.ZipFile(pack_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add README
            readme = f"""# {lesson_title} - Offline Pack

This pack contains {len(words)} signs for offline use.

Contents:
- images/ - Sign images (compressed)
- audio/ - Twi audio files
- worksheet.pdf - Printable worksheet
- index.html - Offline viewer (open in browser)

No internet required after download!

Powered by SignForge • UNICEF Ghana
"""
            zipf.writestr("README.txt", readme)

            # Add sign images (compressed to reduce size)
            for word in words:
                img_path = self.brain_dir / "sign_images" / f"{word.upper()}.png"
                if img_path.exists():
                    # Compress image to 50% quality
                    try:
                        img = Image.open(img_path)
                        # Resize to max 800px width
                        if img.width > 800:
                            ratio = 800 / img.width
                            new_size = (800, int(img.height * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)

                        # Save to temp file
                        temp_path = self.brain_dir / f"temp_{word}.jpg"
                        img.convert('RGB').save(temp_path, 'JPEG', quality=50, optimize=True)

                        zipf.write(temp_path, f"images/{word.lower()}.jpg")
                        temp_path.unlink()  # Delete temp file
                    except Exception:
                        # Fallback: add original image
                        zipf.write(img_path, f"images/{word.lower()}.png")

            # Add audio files if they exist
            from format_creator import get_format_creator
            creator = get_format_creator(self.brain_dir)

            for word in words:
                audio_path = creator.output_dir / "audio" / f"{word.upper().replace(' ', '_')}_AUDIO.mp3"
                if audio_path.exists():
                    zipf.write(audio_path, f"audio/{word.lower()}.mp3")

            # Add PDF worksheet if exists
            pdf_path = creator.output_dir / "pdfs" / f"{lesson_title.replace(' ', '_')}_worksheet.pdf"
            if pdf_path.exists():
                zipf.write(pdf_path, "worksheet.pdf")

            # Create simple HTML viewer
            html_viewer = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{lesson_title} - Ghana Sign Language</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #00549F; }}
        .sign {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .sign img {{ max-width: 100%; height: auto; }}
        audio {{ width: 100%; margin: 10px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; }}
    </style>
</head>
<body>
    <h1>🤟 {lesson_title}</h1>
    <p>Ghana Sign Language - Offline Viewer</p>

"""
            for word in words:
                html_viewer += f"""
    <div class="sign">
        <h2>{word.upper()}</h2>
        <img src="images/{word.lower()}.jpg" alt="{word} sign" onerror="this.src='images/{word.lower()}.png'">
        <audio controls src="audio/{word.lower()}.mp3"></audio>
    </div>
"""

            html_viewer += f"""
    <div class="footer">
        <p>Powered by SignForge • UNICEF Ghana</p>
        <p>No internet required • {len(words)} signs included</p>
    </div>
</body>
</html>
"""
            zipf.writestr("index.html", html_viewer)

        # Get pack size
        pack_size_mb = pack_path.stat().st_size / (1024 * 1024)

        return {
            "success": True,
            "pack_name": pack_name,
            "pack_path": str(pack_path),
            "pack_size_mb": round(pack_size_mb, 2),
            "download_url": f"{self.base_url}/offline_packs/{pack_name}",
            "total_signs": len(words),
            "contents": {
                "images": len(words),
                "audio": len(words),
                "pdf": 1,
                "html_viewer": 1
            }
        }

    # ============================================
    # BULK DELIVERY
    # ============================================

    def send_bulk_sms(self, phones: List[str], word: str) -> Dict:
        """Send SMS to multiple recipients (for village/school broadcasts)"""
        results = []
        successful = 0
        failed = 0

        for phone in phones:
            result = self.send_sms(phone, word)
            results.append(result)

            if result.get('success'):
                successful += 1
            else:
                failed += 1

        return {
            "success": True,
            "total_recipients": len(phones),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    def notify_village_chief(self, phone: str, lesson_title: str, download_url: str) -> Dict:
        """
        Send notification to village chief about new lesson availability
        Chiefs can then distribute to their community
        """
        message = f"📚 New SignForge Lesson: {lesson_title}\n\n"
        message += f"For your community's deaf students\n\n"
        message += f"📥 Download: {self._shorten_url(download_url)}\n\n"
        message += "Share with teachers and parents\n"
        message += "- UNICEF Ghana"

        return self.send_sms(phone, lesson_title)


# Global instance
_rural_service: Optional[RuralDeliveryService] = None

def get_rural_service(brain_dir: Path, config: Optional[Dict] = None) -> RuralDeliveryService:
    """Get or create rural delivery service singleton"""
    global _rural_service
    if _rural_service is None:
        _rural_service = RuralDeliveryService(brain_dir, config)
    return _rural_service
