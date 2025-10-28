"""
TEMPORARY STORAGE SERVICE
Manages generated content with 10-minute TTL
Allows teachers to preview, download, or discard before permanent save
"""

import os
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class TempStorageService:
    """Manages temporary file storage with TTL (Time To Live)"""

    def __init__(self, brain_dir: Path, ttl_minutes: int = 10):
        self.brain_dir = Path(brain_dir)
        self.temp_dir = self.brain_dir / "temp_generated"
        self.temp_dir.mkdir(exist_ok=True)

        self.ttl_seconds = ttl_minutes * 60
        self.metadata_file = self.temp_dir / "temp_metadata.json"

        # Load existing metadata
        self.metadata = self._load_metadata()

        # Clean expired files on init
        self._cleanup_expired()

    def _load_metadata(self) -> Dict:
        """Load metadata about temp files"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _cleanup_expired(self):
        """Remove files older than TTL"""
        current_time = time.time()
        expired_ids = []

        for session_id, data in self.metadata.items():
            created_at = data.get('created_at', 0)
            if current_time - created_at > self.ttl_seconds:
                expired_ids.append(session_id)

        for session_id in expired_ids:
            self.delete_session(session_id)

    def create_session(self, lesson_title: str, words: List[str]) -> str:
        """
        Create a new temporary session for generated content
        Returns session_id for tracking
        """
        # Generate unique session ID
        session_data = f"{lesson_title}_{time.time()}_{len(words)}"
        session_id = hashlib.md5(session_data.encode()).hexdigest()[:16]

        # Create session directory
        session_dir = self.temp_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (session_dir / "qr_codes").mkdir(exist_ok=True)
        (session_dir / "audio").mkdir(exist_ok=True)
        (session_dir / "pdfs").mkdir(exist_ok=True)
        (session_dir / "videos").mkdir(exist_ok=True)

        # Store metadata
        self.metadata[session_id] = {
            "lesson_title": lesson_title,
            "words": words,
            "created_at": time.time(),
            "expires_at": time.time() + self.ttl_seconds,
            "session_dir": str(session_dir),
            "files": []
        }
        self._save_metadata()

        return session_id

    def add_file(self, session_id: str, file_path: str, file_type: str, word: str = None):
        """Add a generated file to session tracking"""
        if session_id not in self.metadata:
            raise ValueError(f"Session {session_id} not found")

        file_info = {
            "path": file_path,
            "type": file_type,
            "word": word,
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }

        self.metadata[session_id]["files"].append(file_info)
        self._save_metadata()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data with time remaining"""
        if session_id not in self.metadata:
            return None

        session = self.metadata[session_id].copy()
        current_time = time.time()
        expires_at = session.get('expires_at', 0)

        # Check if expired
        if current_time > expires_at:
            self.delete_session(session_id)
            return None

        # Add time remaining
        session['time_remaining_seconds'] = int(expires_at - current_time)
        session['time_remaining_display'] = self._format_time_remaining(
            expires_at - current_time
        )

        return session

    def _format_time_remaining(self, seconds: float) -> str:
        """Format time remaining for display"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    def download_session(self, session_id: str) -> Optional[str]:
        """
        Move session files to permanent storage
        Returns path to permanent location
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Create permanent directory
        permanent_dir = self.brain_dir / "generated_formats"
        permanent_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lesson_title = session['lesson_title'].replace(' ', '_')
        permanent_lesson_dir = permanent_dir / f"{lesson_title}_{timestamp}"

        # Copy all files to permanent storage
        session_dir = Path(session['session_dir'])
        shutil.copytree(session_dir, permanent_lesson_dir, dirs_exist_ok=True)

        # Delete temp session
        self.delete_session(session_id)

        return str(permanent_lesson_dir)

    def delete_session(self, session_id: str):
        """Delete session and all its files"""
        if session_id not in self.metadata:
            return

        session_dir = Path(self.metadata[session_id]['session_dir'])

        # Delete directory and all files
        if session_dir.exists():
            shutil.rmtree(session_dir)

        # Remove from metadata
        del self.metadata[session_id]
        self._save_metadata()

    def list_active_sessions(self) -> List[Dict]:
        """List all active (non-expired) sessions"""
        self._cleanup_expired()
        sessions = []

        for session_id in list(self.metadata.keys()):
            session = self.get_session(session_id)
            if session:
                sessions.append({
                    "session_id": session_id,
                    **session
                })

        return sessions

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics about session files"""
        session = self.get_session(session_id)
        if not session:
            return None

        total_size = sum(f['size'] for f in session['files'])
        file_counts = {}

        for file_info in session['files']:
            file_type = file_info['type']
            file_counts[file_type] = file_counts.get(file_type, 0) + 1

        return {
            "session_id": session_id,
            "lesson_title": session['lesson_title'],
            "total_files": len(session['files']),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_counts,
            "time_remaining": session['time_remaining_display'],
            "expires_at": datetime.fromtimestamp(session['expires_at']).isoformat()
        }


# Global instance
_temp_storage: Optional[TempStorageService] = None

def get_temp_storage(brain_dir: Path, ttl_minutes: int = 10) -> TempStorageService:
    """Get or create temp storage singleton"""
    global _temp_storage
    if _temp_storage is None:
        _temp_storage = TempStorageService(brain_dir, ttl_minutes)
    return _temp_storage
