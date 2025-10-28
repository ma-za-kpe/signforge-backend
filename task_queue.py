#!/usr/bin/env python3
"""
Background Task Queue for Auto-Fix Extraction

Handles automatic extraction of missing words in the background
when threshold (2+ user corrections) is met.
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Global extraction queue
extraction_queue = queue.Queue(maxsize=100)
worker_thread: Optional[threading.Thread] = None
should_stop = threading.Event()


def background_extractor_worker(brain_dir: Path):
    """
    Background worker that processes extraction tasks

    Runs continuously in a daemon thread, processing extraction
    requests from the queue.
    """
    logger.info("🚀 Background extraction worker started")

    while not should_stop.is_set():
        try:
            # Get task with timeout to allow checking should_stop
            try:
                task = extraction_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Poison pill
                break

            word = task["word"]
            category = task.get("category", "GENERAL")
            priority = task.get("priority", "medium")

            logger.info(f"🔄 Background auto-fix triggered for '{word}' (priority: {priority})")

            try:
                # Run extraction
                from auto_extractor import get_auto_extractor

                extractor = get_auto_extractor(brain_dir)
                result = extractor.auto_fix_missing_word(word, category)

                # Check if extraction and dictionary update succeeded (even if brain rebuild failed)
                extraction_success = any(
                    step["step"] == "add_to_terms" and step["status"] == "success"
                    for step in result.get("steps", [])
                )

                if extraction_success:
                    logger.info(f"✅ Auto-fix extraction completed for '{word}'")

                    # Mark as processed in correction service (even if brain rebuild failed)
                    # Brain rebuild happens manually via docker exec
                    from correction_service import get_correction_service

                    correction_service = get_correction_service(brain_dir)
                    correction_service.mark_auto_fixed(word)

                    if result["success"]:
                        logger.info(f"🎉 Brain also rebuilt successfully for '{word}'")
                    else:
                        logger.warning(
                            f"⚠️ '{word}' extracted but brain rebuild failed - run manual rebuild"
                        )
                else:
                    logger.error(
                        f"❌ Auto-fix failed for '{word}': {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"❌ Background extraction error for '{word}': {e}")

            extraction_queue.task_done()

        except Exception as e:
            logger.error(f"❌ Worker thread error: {e}")

    logger.info("🛑 Background extraction worker stopped")


def start_background_worker(brain_dir: Path):
    """Start the background extraction worker thread"""
    global worker_thread

    if worker_thread is not None and worker_thread.is_alive():
        logger.warning("⚠️ Background worker already running")
        return

    should_stop.clear()
    worker_thread = threading.Thread(
        target=background_extractor_worker, args=(brain_dir,), daemon=True, name="AutoFixWorker"
    )
    worker_thread.start()
    logger.info("✅ Background extraction worker started")


def stop_background_worker():
    """Stop the background extraction worker thread"""
    global worker_thread

    if worker_thread is None or not worker_thread.is_alive():
        logger.warning("⚠️ No background worker running")
        return

    should_stop.set()
    extraction_queue.put(None)  # Poison pill
    worker_thread.join(timeout=5.0)

    if worker_thread.is_alive():
        logger.warning("⚠️ Worker thread did not stop cleanly")
    else:
        logger.info("✅ Background worker stopped")

    worker_thread = None


def queue_extraction(word: str, category: str = "GENERAL", priority: str = "medium") -> bool:
    """
    Queue a word for background extraction

    Args:
        word: Word to extract
        category: Category (COLORS, FAMILY, etc.)
        priority: Priority level (low, medium, high)

    Returns:
        True if queued successfully, False if queue full
    """
    try:
        task = {"word": word, "category": category, "priority": priority}
        extraction_queue.put_nowait(task)
        logger.info(f"📋 Queued '{word}' for auto-extraction")
        return True
    except queue.Full:
        logger.error(f"❌ Extraction queue full, could not queue '{word}'")
        return False


def get_queue_status() -> Dict:
    """Get current status of the extraction queue"""
    return {
        "queue_size": extraction_queue.qsize(),
        "processing": not extraction_queue.empty(),
        "worker_alive": worker_thread is not None and worker_thread.is_alive(),
        "max_size": extraction_queue.maxsize,
    }
