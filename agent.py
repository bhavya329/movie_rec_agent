"""
Movie & Series Recommendation Agent
-------------------------------------
Fetches new OTT releases (movies AND series) every Friday,
scores them against your taste profile using Gemini AI,
and emails a curated digest — with proper Telugu/Hindi coverage.

Requirements:
    pip install requests google-genai

Environment variables (set as GitHub Actions secrets):
    TMDB_API_KEY       — from themoviedb.org (free)
    GEMINI_API_KEY     — from aistudio.google.com (free)
    GMAIL_ADDRESS      — your Gmail address (sender)
    GMAIL_APP_PASSWORD — Gmail App Password (not your login password)
    RECIPIENT_EMAIL    — where to send the digest
"""

import os
import json
import time
import smtplib
import logging
from collections import Counter
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from google import genai

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User taste profile
# ---------------------------------------------------------------------------
TASTE_PROFILE = {
    "loved_genres":   ["Comedy", "Romance", "Thriller"],
    "hated_genres":   ["Horror", "Documentary"],
    "languages":      ["en", "te", "hi"],
    "language_names": ["English", "Telugu", "Hindi"],
    "min_rating":     5.0,
    "top_n":          6,       # picks in email (2 per language ideally)
}

LANG_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi"}

# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(endpoint: str, params: dict) -> dict:
    params["api_key"] = os.environ["TMDB_API_KEY"]
    resp = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool 1 — Search new MOVIES
# ---------------------------------------------------------------------------
def tool_search_new_movies() -> list[dict]:
    log.info("Tool called: search_new_movies")
    today = datetime.utcnow().date()
    days_ago = today - timedelta(days=45)   # wider window catches more Indian releases
    all_items = []

    # Per-language discover (most reliable for Telugu/Hindi)
    for lang in TASTE_PROFILE["languages"]:
        for page in [1, 2]:   # 2 pages = up to 40 results per language
            data = tmdb_get("/discover/movie", {
                "with_original_language": lang,
                "primary_release_date.gte": str(days_ago),
                "primary_release_date.lte": str(today),
                "sort_by": "release_date.desc",
                "vote_count.gte": 3,
                "page": page,
            })
            items = data.get("results", [])
            if not items:
                break
            log.info("  Movies discover lang=%s page=%d: %d results", lang, page, len(items))
            all_items.extend(items)

    # Weekly trending (catches viral Indian hits)
    data = tmdb_get("/trending/movie/week", {})
    for m in data.get("results", []):
        if m.get("original_language") in TASTE_PROFILE["languages"]:
            all_items.append(m)
    log.info("  Movies trending/week: checked")

    return _dedup(all_items, "movie")


# ---------------------------------------------------------------------------
# Tool 2 — Search new SERIES (TV shows)
# ---------------------------------------------------------------------------
def tool_search_new_series() -> list[dict]:
    log.info("Tool called: search_new_series")
    today = datetime.utcnow().date()
    days_ago = today - timedelta(days=45)
    all_items = []

    for lang in TASTE_PROFILE["languages"]:
        for page in [1, 2]:
            data = tmdb_get("/discover/tv", {
                "with_original_language": lang,
                "first_air_date.gte": str(days_ago),
                "first_air_date.lte": str(today),
                "sort_by": "first_air_date.desc",
                "vote_count.gte": 3,
                "page": page,
            })
            items = data.get("results", [])
            if not items:
                break
            log.info("  Series discover lang=%s page=%d: %d results", lang, page, len(items))
            all_items.extend(items)

    # Weekly trending TV
    data = tmdb_get("/trending/tv/week", {})
    for m in data.get("results", []):
        if m.get("original_language") in TASTE_PROFILE["languages"]:
            all_items.append(m)
    log.info("  Series trending/week: checked")

    return _dedup(all_items, "tv")


def _dedup(items: list[dict], media_type: str) -> list[dict]:
    seen = set()
    unique = []
    for m in items:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            m["media_type"] = media_type
            unique.append(m)
    counts = Counter(m.get("original_language", "?") for m in unique)
    log.info("  Deduped %s: %d unique | %s", media_type, len(unique), dict(counts))
    return unique


# ---------------------------------------------------------------------------
# Tool 3 — Get full details for a movie or TV show
# ---------------------------------------------------------------------------
def tool_get_details(item_id: int, media_type: str) -> dict | None:
    log.info("Tool called: get_details(%d, %s)", item_id, media_type)
    try:
        endpoint = f"/{'movie' if media_type == 'movie' else 'tv'}/{item_id}"
        details = tmdb_get(endpoint, {
            "append_to_response": "watch/providers",
            "language": "en-US",
        })

        # OTT platforms in India
        providers_raw = details.get("watch/providers", {}).get("results", {}).get("IN", {})
        platforms = list({
            p["provider_name"]
            for p in providers_raw.get("flatrate", []) + providers_raw.get("free", [])
        })

        genres = [g["name"] for g in details.get("genres", [])]

        # Normalize movie vs TV fields
        if media_type == "movie":
            title    = details.get("title", "Unknown")
            runtime  = details.get("runtime", 0)
            released = details.get("release_date", "")
        else:
            title    = details.get("name", "Unknown")
            seasons  = details.get("number_of_seasons", 1)
            episodes = details.get("number_of_episodes", 1)
            runtime  = 0   # not applicable for series
            released = details.get("first_air_date", "")

        result = {
            "id":         item_id,
            "media_type": media_type,
            "title":      title,
            "language":   details.get("original_language", ""),
            "overview":   details.get("overview", ""),
            "genres":     genres,
            "rating":     round(details.get("vote_average", 0), 1),
            "vote_count": details.get("vote_count", 0),
            "runtime":    runtime,
            "release":    released,
            "platforms":  platforms,
        }
        if media_type == "tv":
            result["seasons"]  = seasons
            result["episodes"] = episodes
        return result

    except Exception as exc:
        log.warning("  Failed to get details for %d (%s): %s", item_id, media_type, exc)
        return None


# ---------------------------------------------------------------------------
# Tool 4 — Filter by language, rating, and hated genres
# ---------------------------------------------------------------------------
def tool_filter(items: list[dict]) -> list[dict]:
    log.info("Tool called: filter (%d items in)", len(items))
    allowed = set(TASTE_PROFILE["languages"])
    hated   = set(g.lower() for g in TASTE_PROFILE["hated_genres"])

    filtered = []
    for m in items:
        if m.get("original_language") not in allowed:
            continue
        rating = m.get("vote_average", 0) or m.get("rating", 0)
        if rating < TASTE_PROFILE["min_rating"]:
            continue
        filtered.append(m)

    counts = Counter(m.get("original_language", "?") for m in filtered)
    log.info("  After filter: %d items | %s", len(filtered), dict(counts))
    return filtered


def tool_filter_hated_genres(items: list[dict]) -> list[dict]:
    hated = set(g.lower() for g in TASTE_PROFILE["hated_genres"])
    clean = [m for m in items if not any(g.lower() in hated for g in m.get("genres", []))]
    log.info("  After genre filter: %d items remain", len(clean))
    return clean


# ---------------------------------------------------------------------------
# Tool 5 — Gemini AI taste scoring
# ---------------------------------------------------------------------------
def tool_score_taste_match(items: list[dict]) -> list[dict]:
    log.info("Tool called: score_taste_match (%d items)", len(items))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    profile_summary = (
        f"Loves: {', '.join(TASTE_PROFILE['loved_genres'])}. "
        f"Hates: {', '.join(TASTE_PROFILE['hated_genres'])}. "
        f"Languages: {', '.join(TASTE_PROFILE['language_names'])}. "
        f"Wants both movies and series."
    )

    compact = []
    for m in items:
        compact.append({
            "id":         m["id"],
            "type":       m["media_type"],
            "title":      m["title"],
            "genres":     m["genres"],
            "rating":     m.get("rating", m.get("vote_average", 0)),
            "overview":   m.get("overview", "")[:200],
            "language":   m["language"],
        })

    prompt = (
        f"You are a movie/series recommendation agent.\n\n"
        f"USER TASTE: {profile_summary}\n\n"
        f"CONTENT TO SCORE:\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        f"Return ONLY a JSON array, no markdown, no explanation:\n"
        f'[{{"id":<id>,"match_score":<0-100>,"match_reason":"<max 12 words>"}}]'
    )

    raw = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
            )
            raw = response.text.strip()
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                wait = 60 * (attempt + 1)
                log.warning("Gemini quota hit (attempt %d/3). Waiting %ds ...", attempt + 1, wait)
                time.sleep(wait)
            else:
                raise

    if raw is None:
        log.warning("Gemini unavailable. Falling back to genre-based scoring.")
        loved = set(g.lower() for g in TASTE_PROFILE["loved_genres"])
        for m in items:
            overlap = len(loved & set(g.lower() for g in m.get("genres", [])))
            m["match_score"]  = min(50 + overlap * 20, 95)
            m["match_reason"] = "Scored by genre match (AI unavailable)."
        items.sort(key=lambda x: x["match_score"], reverse=True)
        return items

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores    = json.loads(raw)
    score_map = {s["id"]: s for s in scores}

    enriched = []
    for m in items:
        s = score_map.get(m["id"], {"match_score": 0, "match_reason": "No match data."})
        enriched.append({**m, **s})

    enriched.sort(key=lambda x: x["match_score"], reverse=True)
    log.info("Top pick: %s (%s)", enriched[0]["title"] if enriched else "none",
             enriched[0].get("match_score", 0) if enriched else 0)
    return enriched


# ---------------------------------------------------------------------------
# Tool 6 — Pick balanced set (spread across languages)
# ---------------------------------------------------------------------------
def tool_pick_balanced(items: list[dict], n: int) -> list[dict]:
    """
    Ensure the final picks include Telugu and Hindi content,
    not just English. Takes top scoring item per language first,
    then fills remaining slots by score.
    """
    log.info("Tool called: pick_balanced (n=%d)", n)
    by_lang = {"te": [], "hi": [], "en": []}
    for m in items:
        lang = m.get("language", "")
        if lang in by_lang:
            by_lang[lang].append(m)

    picks = []
    # Reserve at least 1 slot for each language that has content
    for lang in ["te", "hi", "en"]:
        if by_lang[lang]:
            picks.append(by_lang[lang][0])

    # Fill remaining slots by overall score
    picked_ids = {m["id"] for m in picks}
    remaining  = [m for m in items if m["id"] not in picked_ids]
    for m in remaining:
        if len(picks) >= n:
            break
        picks.append(m)

    log.info("Balanced picks: %s", [(m["title"], m.get("language")) for m in picks])
    return picks[:n]


# ---------------------------------------------------------------------------
# Email builder
# ---------------------------------------------------------------------------
def _runtime_str(m: dict) -> str:
    if m["media_type"] == "tv":
        s = m.get("seasons", 1)
        e = m.get("episodes", 1)
        return f"{s} season{'s' if s > 1 else ''} · {e} ep{'s' if e > 1 else ''}"
    mins = m.get("runtime", 0)
    if not mins:
        return ""
    h, mi = divmod(mins, 60)
    return f"{h}h {mi}m" if h else f"{mi}m"


def _platform_badges(platforms: list[str]) -> str:
    colors = {
        "Netflix":             ("#E50914", "#fff"),
        "Prime Video":         ("#00A8E1", "#fff"),
        "Disney+ Hotstar":     ("#1D6EC5", "#fff"),
        "Zee5":                ("#8B2FC9", "#fff"),
        "SonyLIV":             ("#D2182B", "#fff"),
        "Apple TV+":           ("#000000", "#fff"),
        "JioCinema":           ("#5C3BC0", "#fff"),
    }
    badges = ""
    for p in platforms[:3]:
        bg, fg = colors.get(p, ("#555", "#fff"))
        badges += (
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:99px;font-size:11px;margin-right:4px;">{p}</span>'
        )
    return badges or '<span style="color:#888;font-size:12px;">Platform TBC</span>'


def _type_badge(media_type: str) -> str:
    if media_type == "tv":
        return '<span style="background:#E1F5EE;color:#085041;font-size:11px;padding:2px 8px;border-radius:99px;margin-right:4px;">Series</span>'
    return '<span style="background:#EEEDFE;color:#3C3489;font-size:11px;padding:2px 8px;border-radius:99px;margin-right:4px;">Movie</span>'


def build_email_html(items: list[dict]) -> str:
    today_str = datetime.utcnow().strftime("%B %d, %Y")
    cards = ""

    for i, m in enumerate(items):
        border   = "border: 1.5px solid #185FA5;" if i == 0 else "border: 1px solid #e0e0e0;"
        top_badge = (
            '<span style="background:#E6F1FB;color:#0C447C;font-size:11px;'
            'padding:2px 8px;border-radius:99px;margin-bottom:8px;display:inline-block;">'
            '⭐ Top pick for you</span><br>'
        ) if i == 0 else ""

        genre_tags = "".join(
            f'<span style="background:#FFF3E0;color:#E65100;font-size:11px;'
            f'padding:2px 8px;border-radius:99px;margin-right:4px;">{g}</span>'
            for g in m.get("genres", [])[:3]
        )
        lang_label = LANG_NAMES.get(m.get("language", ""), m.get("language", "").upper())
        runtime    = _runtime_str(m)
        bar_width  = m.get("match_score", 0)
        bar_color  = "#185FA5" if bar_width >= 70 else "#639922" if bar_width >= 50 else "#BA7517"
        rating     = m.get("rating", m.get("vote_average", 0))

        cards += f"""
<div style="background:#fff;{border}border-radius:12px;padding:16px;margin-bottom:12px;">
  {top_badge}
  <div style="margin-bottom:6px;">
    {_type_badge(m['media_type'])}
    <span style="background:#F5F5F5;color:#555;font-size:11px;padding:2px 8px;border-radius:99px;">{lang_label}</span>
  </div>
  <div style="font-size:17px;font-weight:500;color:#1a1a1a;margin-bottom:4px;">{m['title']}</div>
  <div style="font-size:12px;color:#666;margin-bottom:8px;">⭐ {rating} &nbsp;·&nbsp; {runtime}</div>
  <div style="margin-bottom:8px;">{genre_tags}</div>
  <div style="margin-bottom:10px;">{_platform_badges(m.get('platforms', []))}</div>
  <div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#666;margin-bottom:3px;">
      <span>Taste match</span><span style="font-weight:500;color:#1a1a1a;">{bar_width}%</span>
    </div>
    <div style="height:4px;background:#eee;border-radius:99px;">
      <div style="width:{bar_width}%;height:4px;background:{bar_color};border-radius:99px;"></div>
    </div>
  </div>
  <div style="font-size:12px;color:#666;font-style:italic;">{m.get('match_reason', '')}</div>
</div>"""

    movies_count = sum(1 for m in items if m["media_type"] == "movie")
    series_count = len(items) - movies_count

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;margin:0;padding:24px;">
<div style="max-width:560px;margin:0 auto;">
  <div style="background:#185FA5;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <div style="font-size:22px;font-weight:500;color:#fff;margin-bottom:4px;">🎬 Your Friday Picks</div>
    <div style="font-size:13px;color:#B5D4F4;">{today_str} &nbsp;·&nbsp; Movies &amp; Series · EN / TE / HI</div>
  </div>
  <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 20px;margin-bottom:16px;font-size:14px;color:#444;line-height:1.6;">
    Here are your top <strong>{len(items)} picks</strong> this week —
    <strong>{movies_count} movie{'s' if movies_count != 1 else ''}</strong> and
    <strong>{series_count} series</strong> in English, Telugu &amp; Hindi,
    matched to your taste (Comedy · Romance · Thriller).
  </div>
  {cards}
  <div style="font-size:11px;color:#aaa;text-align:center;margin-top:16px;line-height:1.6;">
    Powered by your Movie Agent · TMDB + Gemini AI · Runs every Friday at 7 PM IST
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tool 7 — Send email
# ---------------------------------------------------------------------------
def tool_send_email(items: list[dict]) -> bool:
    log.info("Tool called: send_email (%d picks)", len(items))

    sender    = os.environ["GMAIL_ADDRESS"].strip()
    recipient = os.environ["RECIPIENT_EMAIL"].strip()
    password  = "".join(c for c in os.environ["GMAIL_APP_PASSWORD"] if c.isascii() and not c.isspace())

    log.info("Sender: %s | Recipient: %s | Password length: %d", sender, recipient, len(password))

    movies_count = sum(1 for m in items if m["media_type"] == "movie")
    series_count = len(items) - movies_count

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎬 Friday picks — {movies_count} movies + {series_count} series matched to you"
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(build_email_html(items), "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        log.info("✅ Email sent to %s", recipient)
        return True
    except smtplib.SMTPAuthenticationError as e:
        log.error("❌ Gmail auth failed: %s", e)
        raise
    except Exception as e:
        log.error("❌ Email error: %s", e)
        raise


# ---------------------------------------------------------------------------
# The Agent — reasoning loop
# ---------------------------------------------------------------------------
def run_agent():
    log.info("=" * 60)
    log.info("Movie & Series Recommendation Agent starting")
    log.info("Goal: Find best new OTT movies AND series in EN/TE/HI and email digest.")
    log.info("=" * 60)

    # Step 1: Search movies and series in parallel
    log.info("Agent reasoning: searching new movies and series separately for better coverage.")
    raw_movies = tool_search_new_movies()
    raw_series = tool_search_new_series()
    all_raw    = raw_movies + raw_series
    log.info("Agent: %d movies + %d series = %d total raw items", len(raw_movies), len(raw_series), len(all_raw))

    if not all_raw:
        log.error("Agent: Nothing found at all. Aborting.")
        return

    # Step 2: Filter by language and rating
    log.info("Agent reasoning: filtering by language (EN/TE/HI) and min rating %.1f", TASTE_PROFILE["min_rating"])
    filtered = tool_filter(all_raw)

    if not filtered:
        log.warning("Agent: nothing after filter — lowering rating bar to 4.0")
        TASTE_PROFILE["min_rating"] = 4.0
        filtered = tool_filter(all_raw)
        if not filtered:
            log.error("Agent: still nothing. Aborting.")
            return

    # Step 3: Fetch full details (genres, platforms, runtime)
    log.info("Agent reasoning: fetching full details for %d candidates.", len(filtered))
    detailed = []
    for m in filtered:
        d = tool_get_details(m["id"], m.get("media_type", "movie"))
        if d:
            detailed.append(d)

    if not detailed:
        log.error("Agent: could not fetch details. Aborting.")
        return

    # Step 4: Remove hated genres
    log.info("Agent reasoning: removing hated genres before AI scoring.")
    clean = tool_filter_hated_genres(detailed)
    if not clean:
        log.warning("Agent: all items in hated genres. Using all detailed items.")
        clean = detailed

    # Step 5: AI taste scoring
    log.info("Agent reasoning: sending %d items to Gemini for taste scoring.", len(clean))
    scored = tool_score_taste_match(clean)

    # Step 6: Pick balanced set across languages
    log.info("Agent reasoning: picking balanced set ensuring TE/HI/EN representation.")
    good   = [m for m in scored if m.get("match_score", 0) >= 35]
    if not good:
        good = scored
    top_picks = tool_pick_balanced(good, TASTE_PROFILE["top_n"])

    # Step 7: Send email
    log.info("Agent reasoning: %d picks ready. Sending email.", len(top_picks))
    tool_send_email(top_picks)
    log.info("Agent: goal achieved. Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    required = ["TMDB_API_KEY", "GEMINI_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
    run_agent()