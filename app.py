import os
import re
import uuid
import json
import base64
import requests
import hashlib
import time

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_file
from openai import OpenAI

load_dotenv()

APP_BOOT_ID = os.getenv("APP_BOOT_ID") or uuid.uuid4().hex

# -------------------- KEYS --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YELP_API_KEY = os.getenv("YELP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not set")

# Optional features - warn but don't fail if not set
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set - translation features disabled")
if not ELEVENLABS_API_KEY:
    print("⚠️  ELEVENLABS_API_KEY not set - TTS features disabled")

# -------------------- APP --------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")


# # dev-safe: works on localhost + adhoc https
# app.config.update(
#     SESSION_COOKIE_SAMESITE="Lax",
#     SESSION_COOKIE_SECURE=False,  # set True only on real HTTPS domain
# )


is_prod = bool(os.getenv("RENDER")) or bool(os.getenv("DYNO")) # set app config based on local/prod env


if is_prod and app.secret_key == "dev-only-change-me":
    raise RuntimeError("Set FLASK_SECRET_KEY in Render/Heroku env vars")


if is_prod:
    app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=True)
else:
    app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=False)


# app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["TTS_CACHE_FOLDER"] = "tts_cache"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["TTS_CACHE_FOLDER"], exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

# -------------------- LANGUAGE CONFIG --------------------
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "yelp_locale": "en_US", "tts_voice": "Rachel"},
    "es": {"name": "Spanish", "yelp_locale": "es_ES", "tts_voice": "Antoni"},
    "fr": {"name": "French", "yelp_locale": "fr_FR", "tts_voice": "Antoni"},
    "de": {"name": "German", "yelp_locale": "de_DE", "tts_voice": "Antoni"},
    "it": {"name": "Italian", "yelp_locale": "it_IT", "tts_voice": "Antoni"},
    "pt": {"name": "Portuguese", "yelp_locale": "pt_BR", "tts_voice": "Antoni"},
    "zh": {"name": "Chinese", "yelp_locale": "zh_CN", "tts_voice": "Rachel"},
    "ja": {"name": "Japanese", "yelp_locale": "ja_JP", "tts_voice": "Rachel"},
    "ko": {"name": "Korean", "yelp_locale": "ko_KR", "tts_voice": "Rachel"},
    "hi": {"name": "Hindi", "yelp_locale": "en_US", "tts_voice": "Rachel"},
    "ar": {"name": "Arabic", "yelp_locale": "en_US", "tts_voice": "Antoni"},
}
DEFAULT_LANGUAGE = "en"

# ElevenLabs voice IDs (multilingual v2 model supports many languages)
ELEVENLABS_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",  # Female, calm
    "Antoni": "ErXwobaYiN019PkySvjV",  # Male, well-rounded
}
ELEVENLABS_MODEL = "eleven_multilingual_v2"

client = OpenAI()

# -------------------- YELP URLS --------------------
YELP_AI_URL = "https://api.yelp.com/ai/chat/v2"
YELP_FUSION_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
YELP_FUSION_BIZ_URL = "https://api.yelp.com/v3/businesses"
YELP_FUSION_REVIEWS_URL = "https://api.yelp.com/v3/businesses/{id}/reviews"

# -------------------- GROUP CHAT (IN-MEMORY) --------------------
# Structure: { group_id: { name, created_at, members: {user_id: {nickname, joined_at}}, messages: [...], restaurants: [...] } }
GROUPS = {}
GROUP_MAX_MESSAGES = 100  # Keep last N messages per group
GROUP_CLEANUP_HOURS = 24  # Auto-delete groups older than this


def generate_group_id():
    """Generate a short, shareable group ID."""
    return uuid.uuid4().hex[:8]


def generate_user_id():
    """Generate a unique user ID for a session."""
    return uuid.uuid4().hex[:12]


def cleanup_old_groups():
    """Remove groups older than GROUP_CLEANUP_HOURS."""
    cutoff = time.time() - (GROUP_CLEANUP_HOURS * 3600)
    to_delete = [gid for gid, g in GROUPS.items() if g.get("created_at", 0) < cutoff]
    for gid in to_delete:
        del GROUPS[gid]


def get_group(group_id):
    """Get group by ID, returns None if not found."""
    return GROUPS.get(group_id)


def create_group(name, creator_nickname):
    """Create a new group and return group_id and creator's user_id."""
    group_id = generate_group_id()
    user_id = generate_user_id()

    GROUPS[group_id] = {
        "name": name,
        "created_at": time.time(),
        "members": {
            user_id: {
                "nickname": creator_nickname,
                "joined_at": time.time(),
                "is_creator": True,
            }
        },
        "messages": [],
        "restaurants": [],
    }

    # Add system message
    GROUPS[group_id]["messages"].append({
        "id": uuid.uuid4().hex[:8],
        "type": "system",
        "text": f"{creator_nickname} created the group",
        "timestamp": time.time(),
    })

    return group_id, user_id


def join_group(group_id, nickname):
    """Join an existing group, returns user_id or None if group not found."""
    group = get_group(group_id)
    if not group:
        return None

    # Check if nickname already taken in this group
    for member in group["members"].values():
        if member["nickname"].lower() == nickname.lower():
            return None  # Nickname taken

    user_id = generate_user_id()
    group["members"][user_id] = {
        "nickname": nickname,
        "joined_at": time.time(),
        "is_creator": False,
    }

    # Add system message
    group["messages"].append({
        "id": uuid.uuid4().hex[:8],
        "type": "system",
        "text": f"{nickname} joined the group",
        "timestamp": time.time(),
    })

    return user_id


def add_group_message(group_id, user_id, text):
    """Add a chat message to a group."""
    group = get_group(group_id)
    if not group or user_id not in group["members"]:
        return None

    member = group["members"][user_id]
    msg = {
        "id": uuid.uuid4().hex[:8],
        "type": "chat",
        "user_id": user_id,
        "nickname": member["nickname"],
        "text": text,
        "timestamp": time.time(),
    }

    group["messages"].append(msg)

    # Trim old messages
    if len(group["messages"]) > GROUP_MAX_MESSAGES:
        group["messages"] = group["messages"][-GROUP_MAX_MESSAGES:]

    return msg


def share_restaurant_to_group(group_id, user_id, restaurant_data):
    """Share a restaurant recommendation to a group."""
    group = get_group(group_id)
    if not group or user_id not in group["members"]:
        return None

    member = group["members"][user_id]

    # Add restaurant to group's list if not already there
    biz_id = restaurant_data.get("business_id") or restaurant_data.get("id")
    existing_ids = [r.get("business_id") for r in group["restaurants"]]

    if biz_id and biz_id not in existing_ids:
        group["restaurants"].append({
            "business_id": biz_id,
            "name": restaurant_data.get("name"),
            "rating": restaurant_data.get("rating"),
            "review_count": restaurant_data.get("review_count"),
            "price": restaurant_data.get("price"),
            "url": restaurant_data.get("url"),
            "image_url": restaurant_data.get("image_url"),
            "categories": restaurant_data.get("categories"),
            "location": restaurant_data.get("location"),
            "shared_by": member["nickname"],
            "shared_at": time.time(),
            "votes": [],  # For future voting feature
        })

    # Add message about the share
    msg = {
        "id": uuid.uuid4().hex[:8],
        "type": "restaurant_share",
        "user_id": user_id,
        "nickname": member["nickname"],
        "restaurant": {
            "business_id": biz_id,
            "name": restaurant_data.get("name"),
            "rating": restaurant_data.get("rating"),
            "price": restaurant_data.get("price"),
            "url": restaurant_data.get("url"),
        },
        "timestamp": time.time(),
    }

    group["messages"].append(msg)
    return msg


@app.before_request
def reset_session_if_server_restarted():
    # If the session was created under a previous server boot, clear the Yelp context
    if session.get("app_boot_id") != APP_BOOT_ID:
        session.clear()  # wipe everything in this session cookie
        session["app_boot_id"] = APP_BOOT_ID

# -------------------- ERROR HANDLER (so frontend always gets JSON) --------------------
@app.errorhandler(Exception)
def handle_any_error(e):
    print("🔥 Server error:", repr(e))
    return jsonify({"error": "server_error", "reply": "Server error."}), 500


# -------------------- HELPERS --------------------
def yelp_headers():
    return {"Authorization": f"Bearer {YELP_API_KEY}", "Accept": "application/json"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def get_user_language():
    """Get the user's selected language from session, default to English."""
    return session.get("language", DEFAULT_LANGUAGE)


def get_yelp_locale():
    """Get the Yelp locale for the user's selected language."""
    lang = get_user_language()
    return SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])["yelp_locale"]


# -------------------- TRANSLATION (GEMINI) --------------------
def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    """
    Translate text using Google's Gemini API.
    Returns original text if translation fails or languages are the same.
    """
    if not GEMINI_API_KEY:
        return text
    if not text or not text.strip():
        return text
    if target_lang == source_lang:
        return text

    target_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang)

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        prompt = f"""Translate the following text to {target_name}.
Only return the translated text, nothing else. Keep formatting (line breaks, emojis, etc.) intact.
If the text contains proper nouns (business names, place names), keep them in their original form.

Text to translate:
{text}"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            }
        }

        resp = requests.post(url, json=payload, timeout=(5, 30))

        if resp.ok:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", text).strip()
        else:
            print(f"Gemini translation error: {resp.status_code} - {resp.text[:500]}")
    except Exception as e:
        print(f"Translation error: {repr(e)}")

    return text


# -------------------- TTS (ELEVENLABS) --------------------
def generate_tts_audio(text: str, lang: str = "en") -> str | None:
    """
    Generate TTS audio using ElevenLabs API.
    Returns the path to the cached audio file, or None if generation fails.
    Uses caching based on text hash to avoid regenerating same audio.
    """
    if not ELEVENLABS_API_KEY:
        return None
    if not text or not text.strip():
        return None

    # Clean text for TTS (remove emojis and special chars that don't vocalize well)
    clean_text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if not clean_text:
        return None

    # Create cache key from text and language
    cache_key = hashlib.md5(f"{clean_text}:{lang}".encode()).hexdigest()
    cache_path = os.path.join(app.config["TTS_CACHE_FOLDER"], f"{cache_key}.mp3")

    # Check cache first
    if os.path.exists(cache_path):
        return cache_path

    # Get voice for language
    voice_name = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])["tts_voice"]
    voice_id = ELEVENLABS_VOICES.get(voice_name, ELEVENLABS_VOICES["Rachel"])

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": clean_text[:5000],  # ElevenLabs has text limits
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=(5, 60))

        if resp.ok:
            with open(cache_path, "wb") as f:
                f.write(resp.content)
            return cache_path
        else:
            print(f"ElevenLabs TTS error: {resp.status_code} - {resp.text[:500]}")
    except Exception as e:
        print(f"TTS generation error: {repr(e)}")

    return None


def cleanup_old_tts_cache(max_age_hours: int = 24):
    """Remove TTS cache files older than max_age_hours."""
    cache_dir = app.config["TTS_CACHE_FOLDER"]
    cutoff = time.time() - (max_age_hours * 3600)

    try:
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
    except Exception as e:
        print(f"Cache cleanup error: {repr(e)}")


# -------------------- STT (GEMINI) --------------------
def transcribe_audio_with_gemini(audio_path: str, lang: str = "en") -> dict:
    """
    Transcribe audio using Google's Gemini API.
    Gemini 2.0 Flash supports native audio input.
    Returns dict with 'text', 'success', and 'detected_language' keys.
    """
    if not GEMINI_API_KEY:
        return {"text": "", "success": False, "error": "STT not configured"}

    if not os.path.exists(audio_path):
        return {"text": "", "success": False, "error": "Audio file not found"}

    try:
        # Read and encode audio file
        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type based on file extension
        ext = audio_path.rsplit(".", 1)[-1].lower()
        mime_types = {
            "webm": "audio/webm",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "m4a": "audio/mp4",
        }
        mime_type = mime_types.get(ext, "audio/webm")

        # Build list of supported language names for detection
        supported_lang_list = ", ".join([f"{code} ({info['name']})" for code, info in SUPPORTED_LANGUAGES.items()])

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        prompt = f"""Transcribe this audio and detect the spoken language.

Supported languages: {supported_lang_list}

Return ONLY valid JSON in this exact format:
{{"text": "transcribed text here", "language": "xx"}}

Where "language" is the 2-letter code (en, es, fr, de, it, pt, zh, ja, ko, hi, ar) of the detected spoken language.
If you cannot understand the audio or it's silent, return: {{"text": "", "language": "en"}}
Do not include any other text or explanation, just the JSON."""

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }

        resp = requests.post(url, json=payload, timeout=(5, 60))

        if resp.ok:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    raw_text = parts[0].get("text", "").strip()

                    # Strip markdown code blocks if present (Gemini often wraps JSON)
                    if raw_text.startswith("```"):
                        # Remove ```json or ``` at start and ``` at end
                        lines = raw_text.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]  # Remove first line
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]  # Remove last line
                        raw_text = "\n".join(lines).strip()

                    # Try to parse as JSON
                    try:
                        result = json.loads(raw_text)
                        text = result.get("text", "").strip()
                        detected_lang = result.get("language", "en").lower()
                        # Validate detected language
                        if detected_lang not in SUPPORTED_LANGUAGES:
                            detected_lang = "en"
                        return {
                            "text": text,
                            "success": True,
                            "detected_language": detected_lang
                        }
                    except json.JSONDecodeError:
                        # Fallback: treat entire response as transcribed text
                        return {
                            "text": raw_text,
                            "success": True,
                            "detected_language": lang  # Keep current language
                        }
            return {"text": "", "success": False, "error": "No transcription returned"}
        else:
            print(f"Gemini STT error: {resp.status_code} - {resp.text[:500]}")
            return {"text": "", "success": False, "error": f"API error: {resp.status_code}"}

    except Exception as e:
        print(f"STT error: {repr(e)}")
        return {"text": "", "success": False, "error": str(e)}


def choose_best_candidate(overlay, vision):
    target = _norm(vision.get("brand_or_business_name") or "") or _norm(
        (vision.get("search_terms") or [""])[0]
    )
    if not target:
        return overlay[0] if overlay else None

    best, best_score = None, -1
    tset = set(target.split())
    for b in overlay:
        name = _norm(b.get("name") or "")
        nset = set(name.split())
        score = len(tset & nset)
        if score > best_score:
            best, best_score = b, score

    return best or (overlay[0] if overlay else None)


def call_yelp_ai(query, lat=None, lon=None, chat_id=None, locale="en_CA"):
    """
    Stable + debuggable.
    Always returns: ({text, raw, status}, new_chat_id)
    """
    payload = {"query": query}
    user_context = {}

    if locale:
        user_context["locale"] = locale
    if lat is not None and lon is not None:
        user_context["latitude"] = float(lat)
        user_context["longitude"] = float(lon)
    if user_context:
        payload["user_context"] = user_context
    if chat_id:
        payload["chat_id"] = chat_id

    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(YELP_AI_URL, headers=headers, json=payload, timeout=(5, 20))
    except requests.RequestException as e:
        print("Yelp AI network error:", repr(e))
        return {"text": "Error talking to Yelp AI (network).", "raw": {}, "status": None}, None

    if not resp.ok:
        print("Yelp AI HTTP:", resp.status_code)
        print("Yelp AI body:", (resp.text or "")[:2000])
        return {"text": f"Yelp AI error ({resp.status_code}).", "raw": {}, "status": resp.status_code}, None

    try:
        data = resp.json()
    except ValueError as e:
        print("Yelp AI JSON decode error:", repr(e))
        print("Yelp AI raw:", (resp.text or "")[:2000])
        return {"text": "Bad response from Yelp AI.", "raw": {}, "status": resp.status_code}, None

    text = ((data.get("response") or {}).get("text")) or "No reply from Yelp AI."
    return {"text": text, "raw": data, "status": resp.status_code}, data.get("chat_id")


def yelp_details(biz_id: str):
    r = requests.get(f"{YELP_FUSION_BIZ_URL}/{biz_id}", headers=yelp_headers(), timeout=(5, 15))
    r.raise_for_status()
    return r.json()


def yelp_reviews(biz_id: str):
    r = requests.get(YELP_FUSION_REVIEWS_URL.format(id=biz_id), headers=yelp_headers(), timeout=(5, 15))
    r.raise_for_status()
    return r.json().get("reviews", [])


def summarize_yelp_reviews(reviews: list) -> dict:
    if not reviews:
        return {
            "overall_sentiment": "No reviews available",
            "highlights": [],
            "complaints": [],
            "one_liner": "No customer reviews found.",
        }

    review_texts = []
    for r in reviews:
        txt = r.get("text")
        rating = r.get("rating")
        if txt:
            review_texts.append(f"[{rating}★] {txt}")

    joined = "\n\n".join(review_texts)

    prompt = f"""
You summarize customer reviews like Amazon.

Return STRICT JSON ONLY with:
- overall_sentiment: string (e.g. "Mostly positive", "Mixed", "Mostly negative")
- highlights: array of 2–4 short positive themes
- complaints: array of 1–3 common negative themes
- one_liner: one sentence summary for customers

Rules:
- Be concise
- Do NOT quote reviews verbatim
- Base only on the provided text

REVIEWS:
{joined}
"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    raw = (resp.output_text or "").strip()

    try:
        return json.loads(raw)
    except Exception:
        return {"overall_sentiment": "Mixed", "highlights": [], "complaints": [], "one_liner": raw[:200]}


def build_business_payload(biz_id: str) -> dict:
    details = yelp_details(biz_id)
    reviews = yelp_reviews(biz_id)
    summary = summarize_yelp_reviews(reviews)

    review_excerpts = []
    for r in (reviews or [])[:3]:
        t = (r.get("text") or "").strip()
        if t:
            review_excerpts.append({"rating": r.get("rating"), "text": t[:220]})

    return {
        "business_id": details.get("id") or biz_id,
        "name": details.get("name"),
        "rating": details.get("rating"),
        "review_count": details.get("review_count"),
        "price": details.get("price"),
        "url": details.get("url"),
        "categories": [c.get("title") for c in (details.get("categories") or []) if isinstance(c, dict)],
        "location": details.get("location"),
        "image_url": details.get("image_url"),
        "review_excerpts": review_excerpts,
        "ai_summary": summary,
    }


def analyze_image_with_vision(image_path: str, lat=None, lon=None) -> str:
    b64 = encode_image(image_path)
    loc_text = ""
    if lat is not None and lon is not None:
        loc_text = (
            f"The approximate GPS location of this photo is: latitude {lat}, longitude {lon}. "
            "Use this to reason about what kind of store or business this could be nearby. "
        )

    prompt = (
        "You are a vision assistant.\n"
        "1. Briefly describe what is in the photo (signs, logos, storefront, interior, etc.).\n"
        "2. Guess the type of business and, if clear, the most likely name.\n"
        "3. If unsure about the exact brand, say so and just describe the business type.\n"
        + loc_text
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
    )
    return resp.output_text or ""


def analyze_live_frame_with_vision(image_path: str, lat=None, lon=None) -> dict:
    b64 = encode_image(image_path)
    loc_hint = f"GPS approx: lat={lat}, lon={lon}." if (lat is not None and lon is not None) else ""

    prompt = f"""
Return STRICT JSON ONLY:
- detected_text: string
- brand_or_business_name: string|null
- business_type: string|null
- confidence: number (0..1)
- search_terms: array of strings (2-6)
- notes: string (max 1 sentence)
{loc_hint}
"""
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
    )
    raw = (resp.output_text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "detected_text": raw[:600],
            "brand_or_business_name": None,
            "business_type": None,
            "confidence": 0.0,
            "search_terms": [raw[:60]] if raw else [],
            "notes": "Vision did not return valid JSON.",
        }


def build_place_context_for_chat() -> str:
    """
    Small stable context block (kept short to avoid token bloat).
    """
    card = session.get("last_place")
    if not isinstance(card, dict):
        return ""

    # Prefer enriched details when available
    enriched = card.get("enriched") if isinstance(card.get("enriched"), dict) else None
    top_pick = card.get("top_pick") if isinstance(card.get("top_pick"), dict) else None

    base = enriched or top_pick
    if not isinstance(base, dict):
        return ""

    name = base.get("name") or ""
    bid = base.get("business_id") or base.get("id") or ""
    url = base.get("url") or ""
    rating = base.get("rating")
    rc = base.get("review_count")
    price = base.get("price")

    parts = [
        "Context: user is asking about THIS place from a recent live scan.",
        f"Name: {name}".strip(),
    ]
    if bid: parts.append(f"Yelp business_id: {bid}")
    if rating is not None or rc is not None:
        parts.append(f"Rating: {rating}  Reviews: {rc}")
    if price: parts.append(f"Price: {price}")
    if url: parts.append(f"URL: {url}")

    # keep it compact
    return "\n".join(parts).strip()


# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/update_location", methods=["POST"])
def update_location():
    data = request.get_json() or {}
    session["lat"] = float(data.get("lat")) if data.get("lat") is not None else None
    session["lon"] = float(data.get("lon")) if data.get("lon") is not None else None
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Please type something 🙂"}), 400

    lat = session.get("lat")
    lon = session.get("lon")
    user_lang = get_user_language()
    yelp_locale = get_yelp_locale()

    # Translate user message to English for Yelp AI if needed
    message_for_yelp = user_message
    if user_lang != "en" and GEMINI_API_KEY:
        message_for_yelp = translate_text(user_message, "en", user_lang)

    # ✅ inject context from last live scan (server-side reliable)
    ctx = build_place_context_for_chat()
    final_message = f"{ctx}\n\nUser: {message_for_yelp}" if ctx else message_for_yelp

    yelp_resp, new_chat_id = call_yelp_ai(
        final_message,
        lat=lat,
        lon=lon,
        chat_id=session.get("yelp_chat_id"),
        locale=yelp_locale,
    )
    if new_chat_id:
        session["yelp_chat_id"] = new_chat_id

    reply_text = yelp_resp["text"]

    # Translate response to user's language if needed
    if user_lang != "en" and GEMINI_API_KEY:
        reply_text = translate_text(reply_text, user_lang, "en")

    return jsonify({
        "reply": reply_text,
        "language": user_lang,
        "meta": {
            "chat_id": session.get("yelp_chat_id"),
            "lat": lat,
            "lon": lon,
            "yelp_status": yelp_resp.get("status"),
            "has_last_place": bool(session.get("last_place")),
        }
    })


@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files.get("image")
    if not f or f.filename == "":
        return jsonify({"error": "No image provided"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported image type. Use jpg/jpeg/png/webp."}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    lat = session.get("lat")
    lon = session.get("lon")
    user_lang = get_user_language()
    yelp_locale = get_yelp_locale()

    try:
        vision_summary = analyze_image_with_vision(save_path, lat=lat, lon=lon)
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass

    yelp_prompt = (
        "Image context from a user photo. You cannot see the image. Use this as context.\n\n"
        f"{vision_summary}\n\n"
        "Infer what kind of business/place this is and provide useful, concise details."
    )

    yelp_resp, new_chat_id = call_yelp_ai(
        yelp_prompt,
        lat=lat,
        lon=lon,
        chat_id=session.get("yelp_chat_id"),
        locale=yelp_locale,
    )
    if new_chat_id:
        session["yelp_chat_id"] = new_chat_id

    reply_text = yelp_resp["text"]

    # Translate response to user's language if needed
    if user_lang != "en" and GEMINI_API_KEY:
        reply_text = translate_text(reply_text, user_lang, "en")

    return jsonify({
        "vision_summary": vision_summary,
        "reply": reply_text,
        "language": user_lang,
        "meta": {"chat_id": session.get("yelp_chat_id"), "lat": lat, "lon": lon},
    })


@app.route("/live_scan", methods=["POST"])
def live_scan():
    f = request.files.get("image")
    if not f or f.filename == "":
        return jsonify({"error": "No image provided"}), 400

    lat = session.get("lat")
    lon = session.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Location missing (allow GPS)."}), 400

    user_lang = get_user_language()
    yelp_locale = get_yelp_locale()

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    try:
        vision = analyze_live_frame_with_vision(save_path, lat=lat, lon=lon)

        terms = vision.get("search_terms") or []
        primary = (vision.get("brand_or_business_name") or "").strip()
        term = primary or (terms[0] if terms else "") or (vision.get("business_type") or "restaurant")

        yelp_query = (
            f"I am pointing my camera at a storefront. Find the most likely matching business for: '{term}'. "
            "Return a short answer and include the best matching businesses nearby with ratings."
        )

        yelp_ai, new_chat_id = call_yelp_ai(
            yelp_query,
            lat=float(lat),
            lon=float(lon),
            chat_id=session.get("yelp_live_chat_id"),
            locale=yelp_locale,
        )
        if new_chat_id:
            session["yelp_live_chat_id"] = new_chat_id

        raw = yelp_ai.get("raw") or {}
        entities = raw.get("entities") or []

        businesses = []
        for e in entities:
            if isinstance(e, dict) and "businesses" in e and isinstance(e["businesses"], list):
                businesses = e["businesses"]
                break

        overlay = []
        for b in (businesses or [])[:5]:
            overlay.append({
                "business_id": b.get("business_id") or b.get("id"),
                "name": b.get("name"),
                "rating": b.get("rating"),
                "review_count": b.get("review_count"),
                "price": b.get("price"),
                "distance_m": b.get("distance"),
                "url": b.get("url"),
                "categories": b.get("categories"),
                "location": b.get("location"),
                "coordinates": b.get("coordinates"),
                "image_url": b.get("image_url"),
            })

        top_pick = choose_best_candidate(overlay, vision)
        enriched = None
        if top_pick and top_pick.get("business_id"):
            try:
                enriched = build_business_payload(top_pick["business_id"])
            except Exception as e:
                print("enrich error:", e)

        # Translate AI text if needed
        yelp_text = yelp_ai.get("text") or ""
        if user_lang != "en" and GEMINI_API_KEY and yelp_text:
            yelp_text = translate_text(yelp_text, user_lang, "en")

        # Translate AI summary one-liner if available
        if enriched and enriched.get("ai_summary") and enriched["ai_summary"].get("one_liner"):
            if user_lang != "en" and GEMINI_API_KEY:
                enriched["ai_summary"]["one_liner"] = translate_text(
                    enriched["ai_summary"]["one_liner"], user_lang, "en"
                )

        return jsonify({
            "mode": "live",
            "vision": vision,
            "yelp_ai": {"text": yelp_text, "status": yelp_ai.get("status"), "candidates": overlay},
            "top_pick": top_pick,
            "enriched": enriched,
            "language": user_lang,
            "meta": {"lat": lat, "lon": lon},
        })

    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass





@app.route("/inject_context", methods=["POST"])
def inject_context():
    data = request.get_json(silent=True) or {}
    card = data.get("card")
    if not isinstance(card, dict):
        return jsonify({"error": "missing_card"}), 400

    enriched = card.get("enriched") if isinstance(card.get("enriched"), dict) else {}
    top = card.get("top_pick") if isinstance(card.get("top_pick"), dict) else {}

    session["last_place"] = {
        "enriched": {
            "business_id": enriched.get("business_id"),
            "name": enriched.get("name"),
            "rating": enriched.get("rating"),
            "review_count": enriched.get("review_count"),
            "price": enriched.get("price"),
            "url": enriched.get("url"),
            "ai_summary": {
                "one_liner": (enriched.get("ai_summary") or {}).get("one_liner")
            },
        } if enriched else None,
        "top_pick": {
            "business_id": top.get("business_id"),
            "name": top.get("name"),
            "rating": top.get("rating"),
            "review_count": top.get("review_count"),
            "price": top.get("price"),
            "url": top.get("url"),
        } if top else None,
    }

    return jsonify({"status": "ok", "stored": True})


# -------------------- LANGUAGE & TTS ENDPOINTS --------------------
@app.route("/get_languages", methods=["GET"])
def get_languages():
    """Return list of supported languages and current selection."""
    current = get_user_language()
    languages = [
        {"code": code, "name": info["name"]}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]
    return jsonify({
        "languages": languages,
        "current": current,
        "translation_enabled": bool(GEMINI_API_KEY),
        "tts_enabled": bool(ELEVENLABS_API_KEY),
        "stt_enabled": bool(GEMINI_API_KEY),  # STT uses Gemini
    })


@app.route("/set_language", methods=["POST"])
def set_language():
    """Set user's preferred language."""
    data = request.get_json(silent=True) or {}
    lang_code = (data.get("language") or "").strip().lower()

    if lang_code not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"Unsupported language: {lang_code}",
            "supported": list(SUPPORTED_LANGUAGES.keys())
        }), 400

    session["language"] = lang_code

    # Clear chat history when changing language for fresh context
    session.pop("yelp_chat_id", None)
    session.pop("yelp_live_chat_id", None)

    return jsonify({
        "status": "ok",
        "language": lang_code,
        "name": SUPPORTED_LANGUAGES[lang_code]["name"]
    })


@app.route("/tts", methods=["POST"])
def generate_tts():
    """Generate TTS audio for given text."""
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "TTS not configured", "tts_enabled": False}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long (max 5000 chars)"}), 400

    lang = data.get("language") or get_user_language()

    # Clean up old cache files periodically (1 in 10 requests)
    if uuid.uuid4().int % 10 == 0:
        cleanup_old_tts_cache()

    audio_path = generate_tts_audio(text, lang)

    if audio_path and os.path.exists(audio_path):
        filename = os.path.basename(audio_path)
        return jsonify({
            "status": "ok",
            "audio_url": f"/tts_audio/{filename}",
            "language": lang,
        })
    else:
        return jsonify({"error": "Failed to generate audio"}), 500


@app.route("/tts_audio/<filename>")
def serve_tts_audio(filename):
    """Serve cached TTS audio files."""
    # Sanitize filename to prevent directory traversal
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400

    if not filename.endswith(".mp3"):
        return jsonify({"error": "Invalid file type"}), 400

    filepath = os.path.join(app.config["TTS_CACHE_FOLDER"], filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "Audio not found"}), 404

    return send_file(filepath, mimetype="audio/mpeg")


@app.route("/stt", methods=["POST"])
def speech_to_text():
    """Convert speech audio to text using Gemini with language detection."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "STT not configured", "stt_enabled": False}), 503

    f = request.files.get("audio")
    if not f or f.filename == "":
        return jsonify({"error": "No audio provided"}), 400

    # Get file extension
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "webm"
    allowed_audio = {"webm", "mp3", "wav", "ogg", "m4a"}
    if ext not in allowed_audio:
        ext = "webm"  # Default to webm for browser recordings

    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)

    user_lang = get_user_language()

    try:
        result = transcribe_audio_with_gemini(save_path, user_lang)

        if result["success"]:
            detected_lang = result.get("detected_language", user_lang)
            language_changed = False

            # Auto-switch language if different from current
            if detected_lang != user_lang and detected_lang in SUPPORTED_LANGUAGES:
                session["language"] = detected_lang
                # Clear chat history for fresh context in new language
                session.pop("yelp_chat_id", None)
                session.pop("yelp_live_chat_id", None)
                language_changed = True

            return jsonify({
                "status": "ok",
                "text": result["text"],
                "detected_language": detected_lang,
                "language_changed": language_changed,
                "language_name": SUPPORTED_LANGUAGES.get(detected_lang, {}).get("name", "English"),
            })
        else:
            return jsonify({
                "error": result.get("error", "Transcription failed"),
                "text": "",
            }), 500
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass


# -------------------- GROUP CHAT ENDPOINTS --------------------
@app.route("/group/create", methods=["POST"])
def create_group_endpoint():
    """Create a new group and return group_id + shareable link."""
    data = request.get_json(silent=True) or {}
    group_name = (data.get("name") or "").strip()
    nickname = (data.get("nickname") or "").strip()

    if not group_name:
        return jsonify({"error": "Group name is required"}), 400
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    if len(group_name) > 50:
        return jsonify({"error": "Group name too long (max 50 chars)"}), 400
    if len(nickname) > 20:
        return jsonify({"error": "Nickname too long (max 20 chars)"}), 400

    # Cleanup old groups periodically
    if uuid.uuid4().int % 5 == 0:
        cleanup_old_groups()

    group_id, user_id = create_group(group_name, nickname)

    # Store user's group membership in session
    session["group_id"] = group_id
    session["group_user_id"] = user_id
    session["group_nickname"] = nickname

    return jsonify({
        "status": "ok",
        "group_id": group_id,
        "user_id": user_id,
        "group_name": group_name,
        "nickname": nickname,
        "share_link": f"/group/{group_id}",
    })


@app.route("/group/<group_id>/join", methods=["POST"])
def join_group_endpoint(group_id):
    """Join an existing group with a nickname."""
    data = request.get_json(silent=True) or {}
    nickname = (data.get("nickname") or "").strip()

    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    if len(nickname) > 20:
        return jsonify({"error": "Nickname too long (max 20 chars)"}), 400

    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = join_group(group_id, nickname)
    if not user_id:
        return jsonify({"error": "Nickname already taken in this group"}), 400

    # Store user's group membership in session
    session["group_id"] = group_id
    session["group_user_id"] = user_id
    session["group_nickname"] = nickname

    return jsonify({
        "status": "ok",
        "group_id": group_id,
        "user_id": user_id,
        "group_name": group["name"],
        "nickname": nickname,
        "member_count": len(group["members"]),
    })


@app.route("/group/<group_id>/info", methods=["GET"])
def group_info(group_id):
    """Get group info (name, members, restaurants)."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    members = [
        {"nickname": m["nickname"], "is_creator": m.get("is_creator", False)}
        for m in group["members"].values()
    ]

    return jsonify({
        "group_id": group_id,
        "name": group["name"],
        "members": members,
        "member_count": len(members),
        "restaurants": group["restaurants"],
        "restaurant_count": len(group["restaurants"]),
    })


@app.route("/group/<group_id>/messages", methods=["GET"])
def get_group_messages(group_id):
    """Poll for new messages. Use ?since=timestamp to get only new messages."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    since = request.args.get("since", type=float, default=0)

    # Filter messages newer than 'since'
    messages = [m for m in group["messages"] if m["timestamp"] > since]

    # Get current user info from session
    current_user_id = session.get("group_user_id")

    return jsonify({
        "messages": messages,
        "count": len(messages),
        "current_user_id": current_user_id,
        "members": [m["nickname"] for m in group["members"].values()],
    })


@app.route("/group/<group_id>/send", methods=["POST"])
def send_group_message(group_id):
    """Send a message to the group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = session.get("group_user_id")
    if not user_id or user_id not in group["members"]:
        return jsonify({"error": "Not a member of this group"}), 403

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Message cannot be empty"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Message too long (max 1000 chars)"}), 400

    msg = add_group_message(group_id, user_id, text)
    if not msg:
        return jsonify({"error": "Failed to send message"}), 500

    return jsonify({"status": "ok", "message": msg})


@app.route("/group/<group_id>/share_restaurant", methods=["POST"])
def share_restaurant_endpoint(group_id):
    """Share a restaurant to the group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = session.get("group_user_id")
    if not user_id or user_id not in group["members"]:
        return jsonify({"error": "Not a member of this group"}), 403

    data = request.get_json(silent=True) or {}
    restaurant = data.get("restaurant")

    if not restaurant or not isinstance(restaurant, dict):
        return jsonify({"error": "Restaurant data required"}), 400

    biz_id = restaurant.get("business_id") or restaurant.get("id")
    if not biz_id:
        return jsonify({"error": "Restaurant must have a business_id"}), 400

    msg = share_restaurant_to_group(group_id, user_id, restaurant)
    if not msg:
        return jsonify({"error": "Failed to share restaurant"}), 500

    return jsonify({
        "status": "ok",
        "message": msg,
        "restaurants": group["restaurants"],
    })


@app.route("/group/<group_id>/restaurants", methods=["GET"])
def get_group_restaurants(group_id):
    """Get list of shared restaurants in the group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    return jsonify({
        "restaurants": group["restaurants"],
        "count": len(group["restaurants"]),
    })


@app.route("/group/<group_id>/leave", methods=["POST"])
def leave_group(group_id):
    """Leave a group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = session.get("group_user_id")
    if not user_id or user_id not in group["members"]:
        return jsonify({"error": "Not a member of this group"}), 403

    nickname = group["members"][user_id]["nickname"]
    del group["members"][user_id]

    # Add leave message
    group["messages"].append({
        "id": uuid.uuid4().hex[:8],
        "type": "system",
        "text": f"{nickname} left the group",
        "timestamp": time.time(),
    })

    # Clear session
    session.pop("group_id", None)
    session.pop("group_user_id", None)
    session.pop("group_nickname", None)

    return jsonify({"status": "ok"})


@app.route("/group/<group_id>/vote", methods=["POST"])
def vote_restaurant(group_id):
    """Vote for a restaurant in the group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = session.get("group_user_id")
    if not user_id or user_id not in group["members"]:
        return jsonify({"error": "Not a member of this group"}), 403

    data = request.get_json(silent=True) or {}
    business_id = (data.get("business_id") or "").strip()

    if not business_id:
        return jsonify({"error": "business_id required"}), 400

    # Find the restaurant
    restaurant = None
    for r in group["restaurants"]:
        if r.get("business_id") == business_id:
            restaurant = r
            break

    if not restaurant:
        return jsonify({"error": "Restaurant not found in group"}), 404

    # Get user's nickname for the vote record
    nickname = group["members"][user_id]["nickname"]

    # Initialize votes list if needed
    if "votes" not in restaurant:
        restaurant["votes"] = []

    # Check if already voted
    existing_vote = None
    for v in restaurant["votes"]:
        if v.get("user_id") == user_id:
            existing_vote = v
            break

    if existing_vote:
        return jsonify({"error": "Already voted for this restaurant"}), 400

    # Add vote
    restaurant["votes"].append({
        "user_id": user_id,
        "nickname": nickname,
        "timestamp": time.time(),
    })

    # Add system message about the vote
    group["messages"].append({
        "id": uuid.uuid4().hex[:8],
        "type": "system",
        "text": f"{nickname} voted for {restaurant['name']}",
        "timestamp": time.time(),
    })

    return jsonify({
        "status": "ok",
        "restaurant": restaurant,
        "vote_count": len(restaurant["votes"]),
        "restaurants": group["restaurants"],
    })


@app.route("/group/<group_id>/unvote", methods=["POST"])
def unvote_restaurant(group_id):
    """Remove vote for a restaurant in the group."""
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    user_id = session.get("group_user_id")
    if not user_id or user_id not in group["members"]:
        return jsonify({"error": "Not a member of this group"}), 403

    data = request.get_json(silent=True) or {}
    business_id = (data.get("business_id") or "").strip()

    if not business_id:
        return jsonify({"error": "business_id required"}), 400

    # Find the restaurant
    restaurant = None
    for r in group["restaurants"]:
        if r.get("business_id") == business_id:
            restaurant = r
            break

    if not restaurant:
        return jsonify({"error": "Restaurant not found in group"}), 404

    # Remove vote
    if "votes" not in restaurant:
        restaurant["votes"] = []

    original_count = len(restaurant["votes"])
    restaurant["votes"] = [v for v in restaurant["votes"] if v.get("user_id") != user_id]

    if len(restaurant["votes"]) == original_count:
        return jsonify({"error": "You haven't voted for this restaurant"}), 400

    return jsonify({
        "status": "ok",
        "restaurant": restaurant,
        "vote_count": len(restaurant["votes"]),
        "restaurants": group["restaurants"],
    })


@app.route("/group/<group_id>")
def group_page(group_id):
    """Render the group chat page."""
    group = get_group(group_id)
    if not group:
        return render_template("index.html", group_error="Group not found")

    return render_template("index.html", group_id=group_id, group_name=group["name"])


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")



if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG") == "1"
    port = int(os.getenv("PORT", "5000"))
    if debug: # run local
        app.run(host="0.0.0.0", port=port, debug=True, ssl_context="adhoc")
    else: # for render/heroku
        app.run(host="0.0.0.0", port=port, debug=False)



