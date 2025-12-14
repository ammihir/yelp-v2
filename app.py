import os
import re
import uuid
import json
import base64
import requests

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI

load_dotenv()

APP_BOOT_ID = os.getenv("APP_BOOT_ID") or uuid.uuid4().hex

# -------------------- KEYS --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YELP_API_KEY = os.getenv("YELP_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not set")

# -------------------- APP --------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")

# dev-safe: works on localhost + adhoc https
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # set True only on real HTTPS domain
)
# app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

client = OpenAI()

# -------------------- YELP URLS --------------------
YELP_AI_URL = "https://api.yelp.com/ai/chat/v2"
YELP_FUSION_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
YELP_FUSION_BIZ_URL = "https://api.yelp.com/v3/businesses"
YELP_FUSION_REVIEWS_URL = "https://api.yelp.com/v3/businesses/{id}/reviews"


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

    # ✅ inject context from last live scan (server-side reliable)
    ctx = build_place_context_for_chat()
    final_message = f"{ctx}\n\nUser: {user_message}" if ctx else user_message

    yelp_resp, new_chat_id = call_yelp_ai(
        final_message,
        lat=lat,
        lon=lon,
        chat_id=session.get("yelp_chat_id"),
        locale="en_CA",
    )
    if new_chat_id:
        session["yelp_chat_id"] = new_chat_id

    return jsonify({
        "reply": yelp_resp["text"],
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
        locale="en_CA",
    )
    if new_chat_id:
        session["yelp_chat_id"] = new_chat_id

    return jsonify({
        "vision_summary": vision_summary,
        "reply": yelp_resp["text"],
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
            locale="en_CA",
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

        return jsonify({
            "mode": "live",
            "vision": vision,
            "yelp_ai": {"text": yelp_ai.get("text"), "status": yelp_ai.get("status"), "candidates": overlay},
            "top_pick": top_pick,
            "enriched": enriched,
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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")













# @app.route("/inject_context", methods=["POST"])
# def inject_context():
#     """
#     ✅ Reliable: store last scanned place server-side
#     (no Yelp call here, so it can’t “break Yelp AI”)
#     """
#     data = request.get_json(silent=True) or {}
#     card = data.get("card")
#     if not isinstance(card, dict):
#         return jsonify({"error": "missing_card"}), 400

#     session["last_place"] = card
#     return jsonify({"status": "ok", "stored": True})
