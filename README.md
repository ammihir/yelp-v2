


access via: http://127.0.0.1:5000/


# TasteLens (Flask + OpenAI Vision + Yelp AI)

TasteLens is a small Flask web app that lets users:
1) chat with Yelp AI using their current location,  
2) upload a photo and ask “what place is this?”, and  
3) do a **Live Scan** (camera frame capture) to identify a nearby business and pull an AI review summary.
4) Group Chat  - allows forming groups and voting on places between members.  

## What it does

### Chat (Yelp AI)
- Frontend sends messages to `/chat`
- Backend calls **Yelp AI Chat v2** with:
  - the user’s message
  - **GPS lat/lon** (if available)
  - a persisted **Yelp chat_id** (so the conversation continues)
- If the user previously did a Live Scan, the backend injects a short “last place” context block into the next chat turn.

### Upload Photo → Vision → Yelp AI
- Frontend uploads an image to `/upload_image`
- Backend:
  1) uses **OpenAI Vision (gpt-4.1-mini)** to describe the image / guess business type/name  
  2) sends that text summary to **Yelp AI** to return useful place details

### Live Scan (camera frame)
- Frontend starts the phone camera, crops the area inside the on-screen frame, and posts it to `/live_scan`
- Backend:
  1) requires GPS (lat/lon must be set)
  2) uses **OpenAI Vision (gpt-4.1-mini)** to return STRICT JSON (name/type/search terms)
  3) calls **Yelp AI** to fetch nearby candidate businesses
  4) picks a “best match” and optionally enriches it using **Yelp Fusion**:
     - Business details endpoint
     - Reviews endpoint
     - OpenAI summarizes reviews into short JSON themes
- The frontend can push this result into chat using `/inject_context` so follow-up questions refer to “THIS place”.

## Tech stack
- Backend: Python + Flask + requests
- AI (vision + review summarization): OpenAI `gpt-4.1-mini`
- Business discovery/chat: Yelp AI Chat v2 + Yelp Fusion (details + reviews)
- Frontend: single HTML page (chat UI + live camera UI)

## Environment variables
Create a `.env` file:

- `OPENAI_API_KEY` (required)
- `YELP_API_KEY` (required)
- `FLASK_SECRET_KEY` (optional; set for real deployments)
- `APP_BOOT_ID` (optional; auto-generated if missing)

## Run locally
```bash
pip install -r requirements.txt
python app.py
