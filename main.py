"""
SignForge Hackathon - Main FastAPI Application
Ghana Sign Language Dictionary API
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try to import dashboard (optional - requires streamlit)
try:
    from analytics_dashboard import get_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠ Analytics dashboard not available")

# Try to import database (optional)
try:
    from database import get_db, init_db, Word, Contribution
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠ Database not available")

# Try to import format creator (optional - requires qrcode, gtts, reportlab)
try:
    from format_creator import get_format_creator
    FORMAT_CREATOR_AVAILABLE = True
except ImportError:
    FORMAT_CREATOR_AVAILABLE = False
    print("⚠ Format creator not available")

# Import contribution system (optional - gracefully handle if not available)
contribution_router = None
try:
    from contribution_api import router as contribution_router
    print("✓ Contribution system loaded")
except ImportError as e:
    print(f"⚠ Contribution system not available: {e}")

# Import training queue system (optional)
training_router = None
try:
    from training_api import router as training_router
    print("✓ AI Training monitoring system loaded")
except ImportError as e:
    try:
        # Fallback to old training_queue if training_api not available
        from training_queue import router as training_router
        print("✓ Training queue system loaded")
    except ImportError as e2:
        print(f"⚠ Training system not available: {e}")

# Import admin AMA system (optional)
admin_router = None
try:
    from admin_ama import router as admin_router
    print("✓ Admin AMA system loaded")
except ImportError as e:
    print(f"⚠ Admin AMA system not available: {e}")

# Import skeleton preview system (optional)
skeleton_router = None
try:
    from skeleton_preview_api import router as skeleton_router
    print("✓ Skeleton preview system loaded")
except ImportError as e:
    print(f"⚠ Skeleton preview system not available: {e}")

# Import reference skeletons system (optional)
reference_skeletons_router = None
try:
    from reference_skeletons_api import router as reference_skeletons_router
    print("✓ Reference skeletons system loaded")
except ImportError as e:
    print(f"⚠ Reference skeletons system not available: {e}")

# Import upload contribution system (video upload feature)
upload_contribution_router = None
try:
    from upload_contribution_api import router as upload_contribution_router
    print("✓ Upload contribution system loaded")
except ImportError as e:
    print(f"⚠ Upload contribution system not available: {e}")

app = FastAPI(
    title="Ghana Sign Language API",
    description="AI-powered sign language search and format generation",
    version="1.0.0",
)

# Initialize database on startup (optional)
@app.on_event("startup")
async def startup_event():
    if DB_AVAILABLE:
        try:
            init_db()
            print("✓ Database initialized")
        except Exception as e:
            print(f"⚠ Database initialization failed: {e}")

    # Preload hybrid search brain to avoid blocking first request
    try:
        from hybrid_search_service import get_hybrid_search_service
        print("🔄 Preloading search brain...")
        search_service = get_hybrid_search_service(BRAIN_DIR)
        print("✅ Search brain preloaded and ready")
    except Exception as e:
        print(f"⚠ Search brain preload failed: {e}")

# CORS configuration for Next.js frontend
# Allow all Vercel deployments and localhost
import re

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-theta-three-66.vercel.app",
        "https://frontend-osplbhh77-popos-projects-fb891440.vercel.app",
        "https://frontend-o9aokhme9-popos-projects-fb891440.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for sign images
# Priority order:
# 1. Railway Volume: /data/ghsl_brain
# 2. Docker: /app/ghsl_brain
# 3. Local dev: ../ghsl_brain
if Path("/data/ghsl_brain").exists():
    BRAIN_DIR = Path("/data/ghsl_brain")
elif Path("/app/ghsl_brain").exists():
    BRAIN_DIR = Path("/app/ghsl_brain")
else:
    BRAIN_DIR = Path(__file__).parent.parent / "ghsl_brain"

if (BRAIN_DIR / "sign_images").exists():
    app.mount(
        "/sign_images", StaticFiles(directory=str(BRAIN_DIR / "sign_images")), name="sign_images"
    )

# Start background auto-fix worker
from task_queue import start_background_worker

start_background_worker(BRAIN_DIR)

# Mount contribution system router (if available)
if contribution_router:
    app.include_router(contribution_router)
    print("✓ Contribution routes mounted")
else:
    print("⚠ Contribution routes not available")

# Mount training queue router (if available)
if training_router:
    app.include_router(training_router)
    print("✓ Training queue routes mounted")
else:
    print("⚠ Training queue routes not available")

# Mount admin AMA router (if available)
if admin_router:
    app.include_router(admin_router)
    print("✓ Admin AMA routes mounted at /api/ama")
else:
    print("⚠ Admin AMA routes not available")

# Mount skeleton preview router (if available)
if skeleton_router:
    app.include_router(skeleton_router)
    print("✓ Skeleton preview routes mounted at /api/skeleton-preview")
else:
    print("⚠ Skeleton preview routes not available")

# Mount reference skeletons router (if available)
if reference_skeletons_router:
    app.include_router(reference_skeletons_router)
    print("✓ Reference skeletons routes mounted at /api/skeletons")
else:
    print("⚠ Reference skeletons routes not available")

# Mount upload contribution router (if available)
if upload_contribution_router:
    app.include_router(upload_contribution_router)
    print("✓ Upload contribution routes mounted at /api/contribute/upload, /api/contribute/submit")
else:
    print("⚠ Upload contribution routes not available")


# Models
class SearchResponse(BaseModel):
    word: str
    sign_image: str
    sign_id: Optional[int] = None
    confidence: float
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    brain_loaded: bool
    total_signs: int


class ContentRequest(BaseModel):
    """Request to extract and normalize words from lesson content"""

    text: str


class ExtractedWord(BaseModel):
    """Word extracted from content with normalization info"""

    original: str
    normalized: Optional[str]
    hasSign: bool
    signImage: Optional[str] = None
    phrase: bool = False
    alternative: bool = False
    reason: Optional[str] = None


class ExtractResponse(BaseModel):
    """Response from content extraction"""

    words: List[ExtractedWord]
    total_words: int
    available_signs: int
    phrases_detected: int


# Endpoints
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Ghana Sign Language API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "search": "/api/search?q=<word>",
            "brain": "/api/brain/stats",
            "formats": "/api/formats/create",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    brain_exists = (BRAIN_DIR / "vectors.index").exists()
    terms_file = BRAIN_DIR / "terms.json"

    total_signs = 0
    if terms_file.exists():
        import json

        with open(terms_file, "r", encoding="utf-8") as f:
            terms = json.load(f)
            total_signs = len(terms)

    return {
        "status": "healthy",
        "version": "1.0.0",
        "brain_loaded": brain_exists,
        "total_signs": total_signs,
    }


@app.get("/api/dictionary-words")
async def get_dictionary_words(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
):
    """
    Get paginated list of dictionary words with contribution status.

    IMPORTANT: Only returns words that are OPEN for contribution (is_open_for_contribution = True).
    This implements the word gating system where admins control which words users can contribute to.

    Returns empty list if no words are currently open.
    """
    if not DB_AVAILABLE:
        # If database is not available, return empty list (no words open)
        return {
            "words": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0
        }

    try:
        # Query database for ONLY open words
        db = next(get_db())
        offset = (page - 1) * per_page

        # Filter only words that are open for contribution
        words = db.query(Word).filter(
            Word.is_open_for_contribution == True
        ).order_by(
            Word.is_complete.asc(),
            Word.contributions_count.desc()
        ).offset(offset).limit(per_page).all()

        total_count = db.query(Word).filter(
            Word.is_open_for_contribution == True
        ).count()

        # Calculate actual contribution counts from source of truth
        from database import Contribution as DBContribution

        words_with_actual_counts = []
        for w in words:
            # Count actual contributions from contributions table (source of truth)
            actual_count = db.query(DBContribution).filter(
                DBContribution.word == w.word
            ).count()

            words_with_actual_counts.append({
                "word": w.word,
                "contributions": actual_count,  # Use ACTUAL count, not stale w.contributions_count
                "needed": w.contributions_needed,
                "ready": actual_count >= w.contributions_needed,  # Calculate from actual
                "quality_score": w.quality_score
            })

        db.close()

        return {
            "words": words_with_actual_counts,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_count + per_page - 1) // per_page if total_count > 0 else 0
        }
    except Exception as e:
        print(f"Error querying open words: {e}")
        # Return empty list on error (fail closed, not open)
        return {
            "words": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0
        }


@app.get("/api/search", response_model=SearchResponse)
async def search_sign(q: str = Query(..., description="Search query (English word)")):
    """
    Search for a sign by English word

    NOW WITH PHRASE NORMALIZATION + HYBRID SEARCH!
    - Phrase normalization ("thank you" → "THANK")
    - Exact match (100%)
    - Fuzzy match (85-98%)
    - Keyword search (60-90%)
    - Semantic vector search (boosted)

    Example: /api/search?q=thank you
    Returns: THANK sign with 100% confidence
    """
    from hybrid_search_service import get_hybrid_search_service
    from phrase_normalizer import get_phrase_normalizer

    # Validate query: empty or whitespace-only should return 404
    if not q or not q.strip():
        raise HTTPException(status_code=404, detail="Query cannot be empty")

    try:
        # STEP 1: Normalize phrase if needed
        normalizer = get_phrase_normalizer(BRAIN_DIR)
        normalized_query, matched_phrase = normalizer.normalize(q)

        # STEP 2: Get HYBRID search service
        search = get_hybrid_search_service(BRAIN_DIR)

        # STEP 3: Search with normalized query
        results = search.search(normalized_query, top_k=10)  # Get top 10 to catch typo matches

        # Validate results: if no results or confidence too low, return 404
        if not results:
            raise HTTPException(status_code=404, detail=f"No sign found for '{q}'")

        # Additional validation: check if query is gibberish
        # Gibberish detection: if no good matches exist in top 5, return 404
        from difflib import SequenceMatcher

        query_normalized = normalized_query.lower()

        # Check similarity of BEST result in top 5 (not just top result)
        best_similarity = 0.0
        best_match_idx = 0

        for idx, result in enumerate(results[:10]):
            matched_word_normalized = result["word"].lower()
            similarity = SequenceMatcher(None, query_normalized, matched_word_normalized).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        # If ALL results have very low similarity (<50%) and no exact match, return 404
        # This catches gibberish like "qwerty" (40% similar to "WORD") and "xyzabc123" (10% similar to "X")
        # But allows typos like "scool" → "SCHOOL" (90% similar)
        if best_similarity < 0.50 and results[0]["confidence"] < 1.0:
            raise HTTPException(status_code=404, detail=f"No sign found for '{q}'")

        # Apply user correction boosting (Human-in-the-Loop learning!)
        from correction_service import get_correction_service

        correction_service = get_correction_service(BRAIN_DIR)
        boosted_results = correction_service.apply_correction_boost(q, results)

        # If we found a much better match by similarity (not just confidence), use that
        # Otherwise use the boosted top result
        if best_match_idx > 0 and best_similarity > 0.80:
            # Use the high-similarity match
            top_result = results[best_match_idx]
        else:
            # Use the top boosted result
            top_result = boosted_results[0]

        # If phrase was normalized, boost confidence to 100%
        if matched_phrase:
            confidence = 1.0
        else:
            confidence = top_result["confidence"]

        # Log search to analytics
        if DASHBOARD_AVAILABLE:
            dashboard = get_dashboard(BRAIN_DIR)
            dashboard.log_search(q, found=True, confidence=confidence)

        return {
            "word": q,
            "sign_image": f"/sign_images/{top_result['image']}",
            "sign_id": top_result.get("page"),
            "confidence": confidence,
            "metadata": {
                "source": "Ghana Sign Language Dictionary 3rd Edition",
                "page": top_result["page"],
                "category": top_result["category"],
                "matched_word": top_result["word"],
                "boosted": top_result.get("boosted", False) or bool(matched_phrase),
                "boost_reason": top_result.get("boost_reason", None)
                or (
                    f'Phrase "{matched_phrase}" normalized to "{normalized_query}"'
                    if matched_phrase
                    else None
                ),
            },
        }

    except HTTPException as e:
        # Log failed searches to analytics
        if e.status_code == 404 and DASHBOARD_AVAILABLE:
            dashboard = get_dashboard(BRAIN_DIR)
            dashboard.log_search(q, found=False, confidence=0.0)

            # LOG MISSING WORD for judges/stakeholders
            from missing_words_tracker import get_missing_tracker

            tracker = get_missing_tracker(BRAIN_DIR)
            tracker.log_missing_word(q)
        # Re-raise HTTP exceptions (404, 400, etc.) as-is
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Brain not loaded - run brain builder first")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-and-normalize", response_model=ExtractResponse)
async def extract_and_normalize_content(request: ContentRequest):
    """
    📝 EXTRACT WORDS FROM LESSON CONTENT AND NORMALIZE

    Takes lesson text, extracts words, applies phrase normalization,
    and returns which words have available signs.

    FIXES CONTENT GENERATION ISSUE: Ensures consistency between
    search (which uses phrase normalization) and content generation.

    Example:
    POST /api/extract-and-normalize
    {
        "text": "Today we learn thank you and four animals"
    }

    Returns:
    {
        "words": [
            {"original": "today", "normalized": "TODAY", "hasSign": true, "signImage": "/sign_images/TODAY.png"},
            {"original": "thank you", "normalized": "THANK", "hasSign": true, "signImage": "/sign_images/THANK.png", "phrase": true},
            {"original": "and", "normalized": "ALSO", "hasSign": true, "signImage": "/sign_images/ALSO.png", "alternative": true},
            {"original": "four", "normalized": "4", "hasSign": true, "signImage": "/sign_images/4.png", "alternative": true}
        ],
        "total_words": 7,
        "available_signs": 7,
        "phrases_detected": 1
    }
    """
    from phrase_normalizer import get_phrase_normalizer
    import re

    try:
        normalizer = get_phrase_normalizer(BRAIN_DIR)
        text = request.text.lower().strip()

        if not text:
            return {
                "words": [],
                "total_words": 0,
                "available_signs": 0,
                "phrases_detected": 0,
            }

        # STRATEGY: Detect phrases first (longest matches win), then process remaining words

        results = []
        text_remaining = text
        detected_phrases = []

        # Phase 1: Detect known multi-word phrases
        # Sort phrases by length (longest first) to match greedily
        phrases_to_check = sorted(normalizer.phrase_map.keys(), key=len, reverse=True)

        for phrase in phrases_to_check:
            # Skip empty-value phrases (natural language patterns)
            if normalizer.phrase_map[phrase] == "":
                continue

            if phrase in text_remaining:
                normalized, _ = normalizer.normalize(phrase)

                # Check if normalized word exists in brain
                if normalized in normalizer.available_words:
                    detected_phrases.append(
                        {
                            "original": phrase,
                            "normalized": normalized,
                            "hasSign": True,
                            "signImage": f"/sign_images/{normalized}.png",
                            "phrase": True,
                        }
                    )
                    # Remove phrase from text to avoid duplicate processing
                    text_remaining = text_remaining.replace(phrase, " ")

        results.extend(detected_phrases)

        # Phase 2: Extract and normalize remaining individual words
        word_matches = re.findall(r"\b[a-z]+\b", text_remaining)
        unique_words = list(dict.fromkeys(word_matches))  # Preserve order, remove duplicates

        for word in unique_words:
            normalized, matched_phrase = normalizer.normalize(word)

            # Check if normalized word exists in brain
            if normalized in normalizer.available_words:
                result = {
                    "original": word,
                    "normalized": normalized,
                    "hasSign": True,
                    "signImage": f"/sign_images/{normalized}.png",
                }

                # Mark if this is an alternative (word mapped to different sign)
                if normalized.upper() != word.upper():
                    result["alternative"] = True
                    result["reason"] = f'"{word}" → "{normalized}"'

                results.append(result)
            else:
                # Word not found
                results.append({"original": word, "normalized": None, "hasSign": False})

        return {
            "words": results,
            "total_words": len(results),
            "available_signs": sum(1 for w in results if w.get("hasSign")),
            "phrases_detected": len(detected_phrases),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content extraction failed: {str(e)}")


@app.get("/api/autocomplete")
async def autocomplete(
    q: str = Query(..., min_length=1, description="Partial query for autocomplete")
):
    """
    Autocomplete/autosuggest endpoint with phrase normalization
    Returns matching words as user types

    Example: /api/autocomplete?q=thank you
    Returns: ["THANK", "THAN", "THINK", ...]
    """
    import json
    from difflib import SequenceMatcher

    from phrase_normalizer import get_phrase_normalizer

    if not q or len(q) < 1:
        return {"suggestions": []}

    try:
        # Get phrase normalizer
        normalizer = get_phrase_normalizer(BRAIN_DIR)

        # Load all words from brain
        metadata_file = BRAIN_DIR / "brain_metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        query_lower = q.lower().strip()
        suggestions = []

        # FIRST: Check if this is a multi-word phrase
        normalized_word, matched_phrase = normalizer.normalize(q)

        if matched_phrase:
            # This is a known phrase! Prioritize the normalized word
            suggestions.append(
                {
                    "word": normalized_word,
                    "match_type": "phrase",
                    "score": 2.0,  # Highest priority
                    "hint": f'(for "{matched_phrase}")',
                }
            )

        # Find direct word matches
        for entry in metadata:
            word = entry.get("word", "")
            word_lower = word.lower()

            # Strategy 1: Starts with query (highest priority)
            if word_lower.startswith(query_lower):
                suggestions.append(
                    {"word": word, "match_type": "prefix", "score": 1.0, "hint": None}
                )
            # Strategy 2: Contains query
            elif query_lower in word_lower:
                suggestions.append(
                    {"word": word, "match_type": "contains", "score": 0.8, "hint": None}
                )
            # Strategy 3: Fuzzy match (for typos)
            else:
                similarity = SequenceMatcher(None, query_lower, word_lower).ratio()
                if similarity > 0.6:  # 60% similar
                    suggestions.append(
                        {"word": word, "match_type": "fuzzy", "score": similarity, "hint": None}
                    )

        # Remove duplicates (keep highest score)
        unique_suggestions = {}
        for s in suggestions:
            word = s["word"]
            if word not in unique_suggestions or s["score"] > unique_suggestions[word]["score"]:
                unique_suggestions[word] = s

        # Sort by score (phrase matches first, then prefix, then contains, then fuzzy)
        sorted_suggestions = sorted(
            unique_suggestions.values(), key=lambda x: x["score"], reverse=True
        )

        # Format response
        result_suggestions = []
        for s in sorted_suggestions[:10]:
            if s["hint"]:
                result_suggestions.append(f"{s['word']} {s['hint']}")
            else:
                result_suggestions.append(s["word"])

        return {
            "query": q,
            "suggestions": result_suggestions,
            "normalized": normalized_word if matched_phrase else None,
            "matched_phrase": matched_phrase,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brain/stats")
async def brain_stats():
    """Get statistics about the vector brain"""
    terms_file = BRAIN_DIR / "terms.json"
    vector_file = BRAIN_DIR / "vectors.index"
    images_dir = BRAIN_DIR / "sign_images"

    stats = {
        "brain_exists": vector_file.exists(),
        "terms_file_exists": terms_file.exists(),
        "total_signs": 0,
        "total_images": 0,
        "brain_size_mb": 0,
    }

    if terms_file.exists():
        import json

        with open(terms_file, "r", encoding="utf-8") as f:
            terms = json.load(f)
            stats["total_signs"] = len(terms)

    if images_dir.exists():
        stats["total_images"] = len(list(images_dir.glob("*.jpg"))) + len(
            list(images_dir.glob("*.png"))
        )

    if vector_file.exists():
        stats["brain_size_mb"] = round(vector_file.stat().st_size / (1024 * 1024), 2)

    return stats


# ============================================
# CORRECTION ENDPOINTS - Human-in-the-Loop
# ============================================


class FlagRequest(BaseModel):
    """Request to flag a search result"""

    query: str
    returned_word: str
    is_correct: bool
    correct_word: Optional[str] = None
    user_comment: Optional[str] = None
    confidence_score: float = 0.0


@app.post("/api/corrections/flag")
async def flag_result(request_data: FlagRequest, request: Request):
    """
    Flag a search result as correct or incorrect

    This enables the human-in-the-loop learning system.
    Users can report when results are wrong and suggest corrections.

    SECURITY: Now tracks IP address and user agent for abuse prevention
    """
    from correction_service import get_correction_service

    # Capture user context for security
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    correction_service = get_correction_service(BRAIN_DIR)

    feedback = correction_service.flag_result(
        query=request_data.query,
        returned_word=request_data.returned_word,
        correct_word=request_data.correct_word,
        is_correct=request_data.is_correct,
        user_comment=request_data.user_comment,
        confidence_score=request_data.confidence_score,
        ip_address=ip_address,  # NEW: Security tracking
        user_agent=user_agent,  # NEW: Security tracking
    )

    return {
        "success": True,
        "feedback_id": feedback["id"],
        "message": "Thank you for your feedback! This helps improve the system.",
        "feedback": feedback,
    }


@app.get("/api/corrections/alternatives")
async def get_alternatives(
    q: str = Query(..., description="Original query to get alternatives for")
):
    """
    Get top 5 alternative matches for a query

    Useful when the top result seems wrong - shows user other options
    """
    from hybrid_search_service import get_hybrid_search_service

    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    try:
        search = get_hybrid_search_service(BRAIN_DIR)
        results = search.search(q, top_k=5)  # Get top 5 instead of just 1

        if not results:
            raise HTTPException(status_code=404, detail=f"No alternatives found for '{q}'")

        # Format results
        alternatives = []
        for idx, result in enumerate(results):
            alternatives.append(
                {
                    "rank": idx + 1,
                    "word": result["word"],
                    "sign_image": f"/sign_images/{result['image']}",
                    "confidence": result["confidence"],
                    "page": result["page"],
                    "category": result["category"],
                }
            )

        return {"query": q, "total_alternatives": len(alternatives), "alternatives": alternatives}

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Brain not loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corrections/stats")
async def correction_stats():
    """
    Get statistics about user corrections and feedback

    Shows how many corrections have been submitted,
    which queries are most problematic, etc.
    """
    from correction_service import get_correction_service

    correction_service = get_correction_service(BRAIN_DIR)
    stats = correction_service.get_correction_stats()

    return stats


@app.get("/api/corrections/for-query")
async def get_query_corrections(q: str = Query(..., description="Query to get corrections for")):
    """
    Get all user corrections for a specific query

    Useful for admin review or understanding problematic queries
    """
    from correction_service import get_correction_service

    correction_service = get_correction_service(BRAIN_DIR)
    corrections = correction_service.get_query_corrections(q)

    return {"query": q, "total_corrections": len(corrections), "corrections": corrections}


# ============================================
# AUTO-FIX ENDPOINTS - Dictionary Maintenance
# ============================================


@app.get("/api/autofix/missing-entries")
async def analyze_missing_entries():
    """
    Analyze user corrections to find missing dictionary entries

    Uses user feedback to identify words that should be in the dictionary
    but are currently missing
    """
    from auto_dictionary_fixer import get_auto_fixer

    fixer = get_auto_fixer(BRAIN_DIR)
    report = fixer.analyze_missing_entries()

    return report


@app.get("/api/autofix/synonym-mappings")
async def get_synonym_mappings():
    """
    Get automatically generated synonym mappings

    Maps user queries to existing dictionary words based on corrections
    """
    from auto_dictionary_fixer import get_auto_fixer

    fixer = get_auto_fixer(BRAIN_DIR)
    synonyms = fixer.auto_create_synonym_mapping()

    return {"total_synonyms": len(synonyms), "synonyms": synonyms}


@app.get("/api/autofix/extraction-targets")
async def get_extraction_targets():
    """
    Get list of words that should be extracted from PDF

    Based on user feedback, generates a prioritized list of signs
    that need to be added to the dictionary
    """
    from auto_dictionary_fixer import get_auto_fixer

    fixer = get_auto_fixer(BRAIN_DIR)
    targets = fixer.generate_extraction_targets()

    return {"total_targets": len(targets), "targets": targets}


@app.get("/api/autofix/similar-words")
async def find_similar_words(
    word: str = Query(..., description="Word to find similar entries for")
):
    """
    Find similar words in dictionary for a missing word

    Helps identify if a word exists under different spelling or synonym
    """
    from auto_dictionary_fixer import get_auto_fixer

    fixer = get_auto_fixer(BRAIN_DIR)
    similar = fixer.suggest_similar_entries(word)

    return {"missing_word": word, "similar_entries": similar}


# ============================================
# AUTO-EXTRACTION - Fully Automatic Fixing
# ============================================


@app.post("/api/autofix/extract-word")
async def auto_extract_word(
    word: str = Query(..., description="Word to automatically extract from PDF"),
    category: str = Query("GENERAL", description="Category for the sign"),
):
    """
    🚀 AUTOMATIC EXTRACTION - The Magic Happens Here!

    When a user reports a missing word, this endpoint:
    1. Finds the PDF automatically
    2. Searches for the word
    3. Extracts the sign image
    4. Adds to dictionary
    5. Rebuilds brain
    6. Returns success!

    Example: POST /api/autofix/extract-word?word=ORANGE&category=COLORS
    """
    from auto_extractor import get_auto_extractor

    try:
        extractor = get_auto_extractor(BRAIN_DIR)
        result = extractor.auto_fix_missing_word(word, category)

        if result["success"]:
            return {
                "success": True,
                "message": f"🎉 Successfully extracted '{word}' from PDF and added to dictionary!",
                "word": word,
                "steps": result["steps"],
                "next_action": f"Try searching for '{word}' again - it should work now!",
            }
        else:
            return {
                "success": False,
                "message": f"❌ Failed to extract '{word}'",
                "word": word,
                "error": result["error"],
                "steps": result["steps"],
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/autofix/queue-status")
async def get_queue_status():
    """
    Check status of background extraction queue

    Shows:
    - Queue size (how many words waiting)
    - Whether worker is processing
    - Worker thread status
    """
    from task_queue import get_queue_status

    status = get_queue_status()

    return {
        "queue_size": status["queue_size"],
        "processing": status["processing"],
        "worker_alive": status["worker_alive"],
        "max_queue_size": status["max_size"],
        "status": "active" if status["worker_alive"] else "inactive",
    }


# ============================================
# FORMAT CREATOR ENDPOINTS - Agent 3
# ============================================


class FormatRequest(BaseModel):
    """Request to create accessibility formats"""

    word: str
    formats: Optional[List[str]] = None  # ["qr", "audio", "pdf", "haptic"]


class LessonBundleRequest(BaseModel):
    """Request to create full lesson bundle"""

    lesson_title: str
    words: List[str]


@app.post("/api/formats/create")
async def create_formats(request: FormatRequest):
    """
    🎨 AGENT 3: FORMAT CREATOR

    Create multiple accessibility formats for a sign word:
    - QR codes (offline access)
    - Twi audio (local language narration)
    - Haptic patterns (vibration for deaf-blind students)

    Example: POST /api/formats/create {"word": "cow", "formats": ["qr", "audio"]}
    """
    from format_creator import get_format_creator

    try:
        creator = get_format_creator(BRAIN_DIR)

        # Get sign image URL
        sign_image_url = f"{request.word.upper()}.png"

        # Create requested formats (or all if none specified)
        if request.formats:
            formats_result = {}
            for fmt in request.formats:
                if fmt == "qr":
                    formats_result["qr_code"] = creator.create_qr_code(request.word, sign_image_url)
                elif fmt == "audio":
                    formats_result["audio"] = creator.create_twi_audio(request.word, request.word)
                elif fmt == "haptic":
                    formats_result["haptic"] = creator.create_haptic_pattern(request.word)

            return {
                "word": request.word,
                "formats": formats_result,
                "total_formats": len(formats_result),
            }
        else:
            # Create all formats
            return creator.create_all_formats(request.word, sign_image_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/formats/lesson-bundle")
async def create_lesson_bundle(request: LessonBundleRequest):
    """
    📚 AGENT 3: CREATE LESSON BUNDLE

    Create complete lesson bundle with all formats for multiple words.
    Perfect for teachers preparing a full lesson.

    Generates:
    - QR codes for each word
    - Twi audio for each word
    - Haptic patterns for each word
    - PDF worksheet with all signs

    Example: POST /api/formats/lesson-bundle
    {
        "lesson_title": "Farm Animals",
        "words": ["cow", "goat", "chicken", "pig"]
    }
    """
    from format_creator import get_format_creator

    try:
        creator = get_format_creator(BRAIN_DIR)
        bundle = creator.create_lesson_bundle(request.words, request.lesson_title)

        # Log lesson creation and format generation to analytics
        if DASHBOARD_AVAILABLE:
            dashboard = get_dashboard(BRAIN_DIR)
            dashboard.log_lesson_creation(request.lesson_title, len(request.words))
            for word in request.words:
                dashboard.log_format_generation(word, ["qr", "audio", "haptic", "pdf"])

        return {
            "success": True,
            "lesson_title": request.lesson_title,
            "total_words": len(request.words),
            "bundle": bundle,
            "message": f"✅ Created lesson bundle with {len(request.words)} signs in multiple formats",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/formats/list/{word}")
async def list_generated_formats(word: str):
    """
    📋 List all generated formats for a word

    Shows what formats have been created and where to find them
    """
    from format_creator import get_format_creator

    creator = get_format_creator(BRAIN_DIR)
    formats_dir = creator.output_dir

    # Check for existing files
    available_formats = {}

    word_upper = word.upper().replace(" ", "_")

    # QR Code
    qr_file = formats_dir / "qr_codes" / f"{word_upper}_QR.png"
    if qr_file.exists():
        available_formats["qr_code"] = str(qr_file)

    # Audio
    audio_file = formats_dir / "audio" / f"{word_upper}_AUDIO.mp3"
    if audio_file.exists():
        available_formats["audio"] = str(audio_file)

    # Haptic
    haptic_file = formats_dir / "videos" / f"{word_upper}_HAPTIC.json"
    if haptic_file.exists():
        available_formats["haptic"] = str(haptic_file)

    return {
        "word": word,
        "available_formats": available_formats,
        "total_formats": len(available_formats),
    }


# ============================================
# ANALYTICS ENDPOINTS - Agent 5
# ============================================


@app.get("/api/metrics")
async def get_metrics():
    """
    📊 AGENT 5: GET ANALYTICS METRICS

    Returns real-time system metrics for analytics dashboard:
    - Total searches and signs generated
    - Search trends and patterns
    - Format generation statistics
    - Regional usage data
    - Impact metrics

    Used by the Streamlit analytics dashboard
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics dashboard not available")

    dashboard = get_dashboard(BRAIN_DIR)
    metrics = dashboard.get_metrics()
    format_stats = dashboard.get_format_stats()

    return {
        "success": True,
        "metrics": metrics,
        "format_stats": format_stats,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/metrics/update")
async def update_metrics(updates: dict):
    """
    📈 AGENT 5: UPDATE ANALYTICS METRICS

    Allows external systems to update analytics metrics
    Useful for tracking offline usage, rural delivery, etc.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics dashboard not available")

    dashboard = get_dashboard(BRAIN_DIR)
    dashboard.update_metrics(updates)

    return {"success": True, "message": "Metrics updated successfully"}


class ContributionAnalyticsEvent(BaseModel):
    """Analytics event for contribution tracking"""
    event_type: str  # "page_view", "word_selected", "contribution_started", "contribution_completed", "contribution_failed"
    word: Optional[str] = None
    metadata: Optional[dict] = None


@app.post("/api/analytics/contribution")
async def log_contribution_analytics(event: ContributionAnalyticsEvent):
    """
    📊 CONTRIBUTION ANALYTICS

    Track contribution system usage for analytics dashboard.

    Event Types:
    - page_view: User views contribution page
    - word_selected: User selects a word to contribute
    - contribution_started: User starts recording
    - contribution_completed: User successfully submits contribution
    - contribution_failed: Contribution submission failed
    """
    if not DASHBOARD_AVAILABLE:
        # Still accept events even if dashboard unavailable (for future processing)
        return {"success": True, "message": "Event logged (dashboard offline)"}

    try:
        dashboard = get_dashboard(BRAIN_DIR)
        metrics = dashboard.get_metrics()

        # Track contribution events
        if "contribution_events" not in metrics:
            metrics["contribution_events"] = {
                "page_views": 0,
                "words_selected": 0,
                "contributions_started": 0,
                "contributions_completed": 0,
                "contributions_failed": 0,
                "by_word": {}
            }

        # Increment event counter
        event_map = {
            "page_view": "page_views",
            "word_selected": "words_selected",
            "contribution_started": "contributions_started",
            "contribution_completed": "contributions_completed",
            "contribution_failed": "contributions_failed"
        }

        if event.event_type in event_map:
            metrics["contribution_events"][event_map[event.event_type]] += 1

        # Track per-word statistics
        if event.word:
            if event.word not in metrics["contribution_events"]["by_word"]:
                metrics["contribution_events"]["by_word"][event.word] = {
                    "selected": 0,
                    "started": 0,
                    "completed": 0,
                    "failed": 0
                }

            word_stats = metrics["contribution_events"]["by_word"][event.word]
            if event.event_type == "word_selected":
                word_stats["selected"] += 1
            elif event.event_type == "contribution_started":
                word_stats["started"] += 1
            elif event.event_type == "contribution_completed":
                word_stats["completed"] += 1
            elif event.event_type == "contribution_failed":
                word_stats["failed"] += 1

        # Update metrics
        dashboard.update_metrics(metrics)

        return {
            "success": True,
            "message": f"Analytics event '{event.event_type}' logged",
            "word": event.word
        }

    except Exception as e:
        # Don't fail the request if analytics fails
        print(f"⚠ Analytics logging failed: {e}")
        return {"success": False, "message": str(e)}


@app.get("/api/analytics/contribution/stats")
async def get_contribution_analytics():
    """
    📊 GET CONTRIBUTION ANALYTICS

    Returns contribution system statistics for dashboard
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics dashboard not available")

    dashboard = get_dashboard(BRAIN_DIR)
    metrics = dashboard.get_metrics()

    # Get contribution events or return defaults
    contribution_events = metrics.get("contribution_events", {
        "page_views": 0,
        "words_selected": 0,
        "contributions_started": 0,
        "contributions_completed": 0,
        "contributions_failed": 0,
        "by_word": {}
    })

    # Calculate conversion rates
    completion_rate = 0
    if contribution_events["contributions_started"] > 0:
        completion_rate = (contribution_events["contributions_completed"] /
                          contribution_events["contributions_started"]) * 100

    # Get top contributed words
    top_words = sorted(
        contribution_events["by_word"].items(),
        key=lambda x: x[1]["completed"],
        reverse=True
    )[:10]

    return {
        "success": True,
        "stats": {
            "overview": {
                "page_views": contribution_events["page_views"],
                "words_selected": contribution_events["words_selected"],
                "contributions_started": contribution_events["contributions_started"],
                "contributions_completed": contribution_events["contributions_completed"],
                "contributions_failed": contribution_events["contributions_failed"],
                "completion_rate": round(completion_rate, 1)
            },
            "top_contributed_words": [
                {
                    "word": word,
                    "completed": stats["completed"],
                    "started": stats["started"],
                    "failed": stats["failed"]
                }
                for word, stats in top_words
            ]
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RURAL DELIVERY ENDPOINTS - Agent 4
# ============================================


class SMSRequest(BaseModel):
    """Request to send SMS"""

    phone: str
    word: str


class WhatsAppRequest(BaseModel):
    """Request to send WhatsApp message"""

    phone: str
    word: str


class USSDRequest(BaseModel):
    """USSD callback request"""

    session_id: str
    service_code: str
    phone_number: str
    text: str


class OfflinePackRequest(BaseModel):
    """Request to create offline pack"""

    lesson_title: str
    words: List[str]


class BulkSMSRequest(BaseModel):
    """Request to send bulk SMS"""

    phones: List[str]
    word: str


@app.post("/api/rural/send-sms")
async def send_rural_sms(request: SMSRequest):
    """
    📱 AGENT 4: SEND SMS

    Send sign language content via SMS to rural areas

    Example: POST /api/rural/send-sms {"phone": "+233123456789", "word": "cow"}
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        result = service.send_sms(request.phone, request.word)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rural/send-whatsapp")
async def send_rural_whatsapp(request: WhatsAppRequest):
    """
    📱 AGENT 4: SEND WHATSAPP

    Send sign language content via WhatsApp to rural areas

    Example: POST /api/rural/send-whatsapp {"phone": "+233123456789", "word": "cow"}
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        result = service.send_whatsapp(request.phone, request.word)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rural/ussd-callback")
async def ussd_callback(request: USSDRequest):
    """
    📱 AGENT 4: USSD CALLBACK

    Handle USSD menu navigation for feature phones (*123#)

    This endpoint is called by Africa's Talking when user dials *123#

    Returns USSD response in AT format:
    - "CON ..." for continue (show menu)
    - "END ..." for end session (final message)
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        response_text, end_session = service.handle_ussd_request(
            request.session_id, request.service_code, request.phone_number, request.text
        )

        return {"response": response_text, "end_session": end_session}

    except Exception as e:
        # Return error in USSD format
        return {
            "response": f"END ⚠️ Error: {str(e)}\n\nPlease try again later.",
            "end_session": True,
        }


@app.post("/api/rural/whatsapp-webhook")
async def whatsapp_webhook(From: str = "", Body: str = ""):
    """
    📱 AGENT 4: WHATSAPP WEBHOOK

    Handle incoming WhatsApp messages

    This endpoint is called by Twilio when user sends WhatsApp message

    Supported commands:
    - "cow" → Send COW sign
    - "LESSON" → List lessons
    - "HELP" → Help message
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        result = service.handle_whatsapp_webhook(Body, From)

        return result

    except Exception as e:
        return {"response": f"❌ Error: {str(e)}", "action": "error"}


@app.post("/api/rural/offline-pack")
async def create_offline_pack(request: OfflinePackRequest):
    """
    📥 AGENT 4: CREATE OFFLINE PACK

    Create downloadable offline pack for rural areas with limited connectivity

    Pack includes:
    - Compressed sign images
    - Audio files
    - PDF worksheet
    - HTML offline viewer

    Optimized for 2G/3G networks

    Example: POST /api/rural/offline-pack
    {"lesson_title": "Farm Animals", "words": ["cow", "goat", "chicken"]}
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        result = service.create_offline_pack(request.lesson_title, request.words)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rural/bulk-sms")
async def send_bulk_sms(request: BulkSMSRequest):
    """
    📱 AGENT 4: SEND BULK SMS

    Send SMS to multiple recipients (village/school broadcasts)

    Example: POST /api/rural/bulk-sms
    {"phones": ["+233123456789", "+233987654321"], "word": "cow"}
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)
        result = service.send_bulk_sms(request.phones, request.word)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rural/delivery-log")
async def get_delivery_log():
    """
    📊 AGENT 4: GET DELIVERY LOG

    Returns delivery statistics and history
    """
    from rural_delivery import get_rural_service

    try:
        service = get_rural_service(BRAIN_DIR)

        # Load delivery log
        import json

        with open(service.delivery_log_file, "r") as f:
            log = json.load(f)

        return {"success": True, "log": log}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# TEMPORARY STORAGE & MISSING WORD FINDER
# ============================================


class TempLessonRequest(BaseModel):
    """Request to create temporary lesson"""

    lesson_title: str
    words: List[str]
    find_missing: bool = True  # Auto-search for missing words


class SessionActionRequest(BaseModel):
    """Request to download or discard session"""

    session_id: str


@app.post("/api/temp/create-lesson")
async def create_temp_lesson(request: TempLessonRequest):
    """
    📝 CREATE TEMPORARY LESSON (10min TTL)

    Creates lesson with all formats in temporary storage.
    Teacher can preview, then download or discard.
    Automatically searches for missing words if enabled.

    Example:
    POST /api/temp/create-lesson
    {
        "lesson_title": "Farm Animals",
        "words": ["cow", "goat", "some", "of"],
        "find_missing": true
    }
    """
    from format_creator import get_format_creator
    from missing_word_finder import get_missing_word_finder
    from temp_storage_service import get_temp_storage

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        format_creator = get_format_creator(BRAIN_DIR)

        # Create temp session
        session_id = temp_storage.create_session(request.lesson_title, request.words)

        # Find missing words if enabled
        missing_results = None
        if request.find_missing:
            missing_finder = get_missing_word_finder(BRAIN_DIR)
            missing_results = missing_finder.batch_find_missing(request.words)

        # Generate formats for all words (in temp directory)
        session_dir = Path(temp_storage.metadata[session_id]["session_dir"])

        # Temporarily override output_dir
        original_output_dir = format_creator.output_dir
        format_creator.output_dir = session_dir

        # Recreate subdirectories with temp paths
        format_creator.output_dir.mkdir(exist_ok=True)
        (format_creator.output_dir / "qr_codes").mkdir(exist_ok=True)
        (format_creator.output_dir / "audio").mkdir(exist_ok=True)
        (format_creator.output_dir / "pdfs").mkdir(exist_ok=True)
        (format_creator.output_dir / "videos").mkdir(exist_ok=True)

        # Create lesson bundle in temp directory
        bundle = format_creator.create_lesson_bundle(request.words, request.lesson_title)

        # Restore original output_dir
        format_creator.output_dir = original_output_dir

        # Track files in temp storage
        for word_format in bundle.get("word_formats", []):
            if "formats" in word_format:
                for fmt_type, fmt_data in word_format["formats"].items():
                    if "file_path" in fmt_data:
                        temp_storage.add_file(
                            session_id, fmt_data["file_path"], fmt_type, word_format.get("word")
                        )

        # Add PDF worksheet
        if "pdf_worksheet" in bundle and "file_path" in bundle["pdf_worksheet"]:
            temp_storage.add_file(session_id, bundle["pdf_worksheet"]["file_path"], "pdf", None)

        # Get session stats
        session_stats = temp_storage.get_session_stats(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "session_stats": session_stats,
            "missing_words": missing_results,
            "preview_url": f"/api/temp/preview/{session_id}",
            "message": f"✅ Temporary lesson created. Expires in 10 minutes. Preview and download or discard.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/temp/preview/{session_id}")
async def preview_temp_lesson(session_id: str):
    """
    👁️ PREVIEW TEMPORARY LESSON

    View all generated files before downloading
    """
    from temp_storage_service import get_temp_storage

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        session = temp_storage.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        stats = temp_storage.get_session_stats(session_id)

        return {
            "session": session,
            "stats": stats,
            "actions": {
                "download": f"/api/temp/download/{session_id}",
                "discard": f"/api/temp/discard/{session_id}",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/temp/download-zip/{session_id}")
async def download_lesson_zip(session_id: str):
    """
    ⬇️ DOWNLOAD LESSON AS ZIP FILE

    Returns a ZIP file containing all generated files for client-side download
    """
    from fastapi.responses import FileResponse
    from temp_storage_service import get_temp_storage
    import zipfile
    import tempfile

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        session_data = temp_storage.get_session(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        # Create a temporary ZIP file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as zip_temp:
            zip_path = zip_temp.name

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from the session
                for file_entry in session_data.get('files', []):
                    file_path = Path(file_entry['path'])
                    if file_path.exists():
                        # Use word as folder name for organization
                        word = file_entry.get('word', 'file')
                        file_type = file_entry.get('type', 'unknown')
                        arcname = f"{word}/{file_type}/{file_path.name}"
                        zipf.write(file_path, arcname=arcname)

        # Return the ZIP file with proper headers for download
        lesson_name = session_data.get('lesson_name', 'lesson')
        filename = f"{lesson_name.replace(' ', '_')}_materials.zip"

        return FileResponse(
            path=zip_path,
            media_type='application/zip',
            filename=filename,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/temp/download/{session_id}")
async def download_temp_lesson(session_id: str):
    """
    ⬇️ SAVE TO SERVER STORAGE (Legacy)

    Move files from temp to permanent storage on server
    """
    from temp_storage_service import get_temp_storage

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        permanent_path = temp_storage.download_session(session_id)

        if not permanent_path:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        return {
            "success": True,
            "message": "Lesson saved to server storage",
            "permanent_path": permanent_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/temp/discard/{session_id}")
async def discard_temp_lesson(session_id: str):
    """
    🗑️ DISCARD TEMPORARY LESSON

    Delete all temporary files immediately
    """
    from temp_storage_service import get_temp_storage

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        temp_storage.delete_session(session_id)

        return {"success": True, "message": "Temporary lesson discarded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/temp/list")
async def list_temp_sessions():
    """
    📋 LIST ALL ACTIVE TEMPORARY SESSIONS
    """
    from temp_storage_service import get_temp_storage

    try:
        temp_storage = get_temp_storage(BRAIN_DIR)
        sessions = temp_storage.list_active_sessions()

        return {"success": True, "sessions": sessions, "total": len(sessions)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/missing/find")
async def find_missing_word(word: str):
    """
    🔍 FIND MISSING WORD

    Search open-source databases for a word not in our brain
    """
    from missing_word_finder import get_missing_word_finder

    try:
        finder = get_missing_word_finder(BRAIN_DIR)
        result = finder.find_missing_word(word)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/missing/batch-find")
async def batch_find_missing(words: List[str]):
    """
    🔍 BATCH FIND MISSING WORDS

    Search for multiple missing words at once
    """
    from missing_word_finder import get_missing_word_finder

    try:
        finder = get_missing_word_finder(BRAIN_DIR)
        results = finder.batch_find_missing(words)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# MISSING WORDS TRACKING & REPORTING
# ============================================


@app.get("/api/missing/report")
async def get_missing_words_report():
    """
    📊 GET MISSING WORDS REPORT

    Returns comprehensive report of all words teachers searched for but we don't have.
    Perfect for showing judges and planning future sign additions.
    """
    from missing_words_tracker import get_missing_tracker

    try:
        tracker = get_missing_tracker(BRAIN_DIR)
        stats = tracker.get_stats()

        return {
            "success": True,
            "stats": stats,
            "report_file": str(tracker.missing_words_report),
            "csv_file": str(tracker.missing_words_csv),
            "json_file": str(tracker.missing_words_json),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/missing/top")
async def get_top_missing_words(limit: int = 20):
    """
    🔝 GET TOP MISSING WORDS

    Returns most frequently requested missing words
    """
    from missing_words_tracker import get_missing_tracker

    try:
        tracker = get_missing_tracker(BRAIN_DIR)
        top_words = tracker.get_top_missing(limit)

        return {"success": True, "limit": limit, "top_missing_words": top_words}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/reset-and-seed-database")
async def reset_and_seed_database():
    """ADMIN ONLY: Reset database and reseed with correct sign words from brain_metadata.json"""
    import json

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        from database import SessionLocal, Word, init_db

        # Initialize database
        init_db()
        db = SessionLocal()

        # Delete all existing words
        deleted = db.query(Word).delete()
        db.commit()

        # Load brain metadata (EXACT SAME LOGIC as seed_database.py)
        brain_file = Path("/data/ghsl_brain/brain_metadata.json")
        if not brain_file.exists():
            brain_file = BRAIN_DIR / "brain_metadata.json"
        if not brain_file.exists():
            brain_file = Path("../ghsl_brain/brain_metadata.json")

        if not brain_file.exists():
            raise HTTPException(status_code=404, detail="brain_metadata.json not found")

        with open(brain_file, 'r') as f:
            brain_data = json.load(f)

        # Handle if brain_data is a list or dict (EXACT SAME as seed_database.py)
        if isinstance(brain_data, list):
            words_list = [item['word'].upper() for item in brain_data if 'word' in item]
        else:
            words_list = [sign_data['word'].upper() for sign_id, sign_data in brain_data.items()]

        # Deduplicate words_list
        words_list = list(set(words_list))

        # Insert new words (skip duplicates)
        added = 0
        skipped = 0
        for idx, word in enumerate(words_list, 1):
            try:
                # Check if word already exists
                existing = db.query(Word).filter(Word.word == word).first()
                if existing:
                    skipped += 1
                    continue

                word_entry = Word(
                    word=word,
                    contributions_count=0,
                    contributions_needed=50,
                    is_complete=False,
                    quality_score=None
                )
                db.add(word_entry)
                db.commit()  # Commit each word individually to avoid batch conflicts
                added += 1
            except Exception as e:
                db.rollback()
                skipped += 1
                print(f"Skipped {word}: {e}")

        total_words = db.query(Word).count()
        first_10 = [w.word for w in db.query(Word).limit(10).all()]

        db.close()

        return {
            "success": True,
            "deleted": deleted,
            "added": added,
            "skipped": skipped,
            "total_in_db": total_words,
            "sample_words": first_10
        }

    except Exception as e:
        if 'db' in locals():
            db.rollback()
            db.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/corrections/reset")
async def reset_user_corrections():
    """ADMIN ONLY: Reset all user corrections by deleting the user_corrections.json file"""
    import json

    corrections_file = BRAIN_DIR / "user_corrections.json"

    if not corrections_file.exists():
        return {
            "success": True,
            "message": "No user corrections file found (already clean)"
        }

    try:
        # Backup the file before deleting
        backup_file = BRAIN_DIR / f"user_corrections_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Read current corrections
        with open(corrections_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Save backup
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Delete the corrections file
        corrections_file.unlink()

        correction_count = len(data.get("corrections", []))

        return {
            "success": True,
            "message": f"Deleted {correction_count} user corrections",
            "backup_file": str(backup_file),
            "deleted_corrections": correction_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset corrections: {str(e)}")


@app.delete("/api/admin/corrections/{query}")
async def delete_query_correction(query: str):
    """ADMIN ONLY: Delete user correction for a specific query"""
    import json

    corrections_file = BRAIN_DIR / "user_corrections.json"

    if not corrections_file.exists():
        raise HTTPException(status_code=404, detail="No user corrections file found")

    try:
        # Read current corrections
        with open(corrections_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        corrections = data.get("corrections", [])
        original_count = len(corrections)

        # Filter out corrections for this query
        query_lower = query.lower()
        filtered_corrections = [c for c in corrections if c.get("query") != query_lower]
        deleted_count = original_count - len(filtered_corrections)

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"No corrections found for query: {query}")

        # Update data
        data["corrections"] = filtered_corrections
        data["total_corrections"] = len(filtered_corrections)
        data["last_updated"] = datetime.now().isoformat()

        # Save updated file
        with open(corrections_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"Deleted {deleted_count} correction(s) for query: {query}",
            "remaining_corrections": len(filtered_corrections)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete corrections: {str(e)}")


@app.post("/api/admin/migrate-classification-fields")
async def migrate_classification_fields():
    """ADMIN ONLY: Add classification and 3-attempt fields to database"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        from sqlalchemy import text, create_engine
        import os

        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghsl:ghsl_dev_pass@localhost:5432/ghsl_contributions")
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            # Add fields to Word table
            conn.execute(text("""
                ALTER TABLE words
                ADD COLUMN IF NOT EXISTS static_votes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS dynamic_votes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS sign_type_consensus VARCHAR(20),
                ADD COLUMN IF NOT EXISTS consensus_confidence FLOAT;
            """))
            conn.commit()

            # Add fields to Contribution table
            conn.execute(text("""
                ALTER TABLE contributions
                ADD COLUMN IF NOT EXISTS sign_type_movement VARCHAR(20),
                ADD COLUMN IF NOT EXISTS sign_type_hands VARCHAR(20),
                ADD COLUMN IF NOT EXISTS num_attempts INTEGER DEFAULT 1,
                ADD COLUMN IF NOT EXISTS individual_qualities JSON,
                ADD COLUMN IF NOT EXISTS individual_durations JSON,
                ADD COLUMN IF NOT EXISTS quality_variance FLOAT,
                ADD COLUMN IF NOT EXISTS improvement_trend VARCHAR(100);
            """))
            conn.commit()

        return {
            "success": True,
            "message": "Migration successful: added classification and 3-attempt fields"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


@app.post("/api/admin/migrate-file-path-nullable")
async def migrate_file_path_nullable():
    """ADMIN ONLY: Make file_path column nullable in contributions table"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        from sqlalchemy import text, create_engine
        import os

        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghsl:ghsl_dev_pass@localhost:5432/ghsl_contributions")
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            # Make file_path nullable
            conn.execute(text("""
                ALTER TABLE contributions
                ALTER COLUMN file_path DROP NOT NULL
            """))
            conn.commit()

        return {
            "success": True,
            "message": "Migration successful: file_path is now nullable"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
