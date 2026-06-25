"""
Movie Recommendation Agent
--------------------------
An agentic AI system that fetches new OTT releases every Friday,
scores them against your taste profile using Gemini, and emails
a curated digest to your inbox.

Requirements:
    pip install requests google-genai

Environment variables (set as GitHub Actions secrets):
    TMDB_API_KEY       — from themoviedb.org (free)
    GEMINI_API_KEY     — from aistudio.google.com (free)
    GMAIL_ADDRESS      — your Gmail address (sender)
    GMAIL_APP_PASSWORD — Gmail App Password (not your login password)
    RECIPIENT_EMAIL    — where to send the digest (can be same as above)
"""

import os
import json
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import time
from google import genai

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User taste profile — edit this to match your preferences
# ---------------------------------------------------------------------------
TASTE_PROFILE = {
    "loved_genres":   ["Comedy", "Romance", "Thriller"],
    "hated_genres":   ["Horror", "Documentary"],
    "languages":      ["en", "te", "hi"],          # ISO 639-1 codes
    "language_names": ["English", "Telugu", "Hindi"],
    "format":         "Movies only",
    "min_rating":     5.5,                          # skip anything below this TMDB score
    "top_n":          5,                            # number of picks to include in email
}

# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(endpoint: str, params: dict) -> dict:
    """Make a GET request to TMDB and return JSON."""
    params["api_key"] = os.environ["TMDB_API_KEY"]
    resp = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def tool_search_new_releases() -> list[dict]:
    """
    Tool: Fetch movies released in the last 14 days on OTT in India.
    Returns a list of raw TMDB movie objects.
    """
    log.info("Tool called: search_new_releases")
    today = datetime.utcnow().date()
    two_weeks_ago = today - timedelta(days=14)

    all_movies = []
    for lang in TASTE_PROFILE["languages"]:
        data = tmdb_get("/discover/movie", {
            "with_original_language": lang,
            "primary_release_date.gte": str(two_weeks_ago),
            "primary_release_date.lte": str(today),
            "region": "IN",
            "sort_by": "popularity.desc",
            "with_watch_monetization_types": "flatrate|free",
            "watch_region": "IN",
            "page": 1,
        })
        movies = data.get("results", [])
        log.info("  Found %d movies for language '%s'", len(movies), lang)
        all_movies.extend(movies)

    # Deduplicate by movie id
    seen = set()
    unique = []
    for m in all_movies:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)

    log.info("Total unique movies found: %d", len(unique))
    return unique


def tool_get_movie_details(movie_id: int) -> dict:
    """
    Tool: Get full details for a movie — genres, runtime, tagline, OTT platforms.
    Returns enriched movie dict or None on failure.
    """
    log.info("Tool called: get_movie_details(%d)", movie_id)
    try:
        details = tmdb_get(f"/movie/{movie_id}", {
            "append_to_response": "watch/providers",
            "language": "en-US",
        })

        # Extract OTT platforms available in India
        providers_raw = details.get("watch/providers", {}).get("results", {}).get("IN", {})
        flatrate = providers_raw.get("flatrate", [])
        free     = providers_raw.get("free", [])
        platforms = list({p["provider_name"] for p in flatrate + free})

        genres = [g["name"] for g in details.get("genres", [])]

        return {
            "id":          movie_id,
            "title":       details.get("title", "Unknown"),
            "language":    details.get("original_language", ""),
            "overview":    details.get("overview", ""),
            "genres":      genres,
            "rating":      round(details.get("vote_average", 0), 1),
            "vote_count":  details.get("vote_count", 0),
            "runtime":     details.get("runtime", 0),
            "tagline":     details.get("tagline", ""),
            "release":     details.get("release_date", ""),
            "platforms":   platforms,
            "poster":      details.get("poster_path", ""),
        }
    except Exception as exc:
        log.warning("  Failed to get details for %d: %s", movie_id, exc)
        return None


def tool_filter_by_language(movies: list[dict]) -> list[dict]:
    """
    Tool: Keep only movies in the user's preferred languages and
    above the minimum rating threshold.
    """
    log.info("Tool called: filter_by_language (%d movies in)", len(movies))
    allowed = set(TASTE_PROFILE["languages"])
    filtered = [
        m for m in movies
        if m.get("original_language") in allowed
        and m.get("vote_average", 0) >= TASTE_PROFILE["min_rating"]
    ]
    log.info("  %d movies remain after language/rating filter", len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Gemini AI helper
# ---------------------------------------------------------------------------

def tool_score_taste_match(movies: list[dict]) -> list[dict]:
    """
    Tool: Ask Gemini to score and reason about each movie against the
    user's taste profile. Returns movies sorted by match score descending.
    """
    log.info("Tool called: score_taste_match (%d movies)", len(movies))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    profile_summary = (
        f"The user loves: {', '.join(TASTE_PROFILE['loved_genres'])}. "
        f"They hate: {', '.join(TASTE_PROFILE['hated_genres'])}. "
        f"They prefer {TASTE_PROFILE['format']} in "
        f"{', '.join(TASTE_PROFILE['language_names'])}."
    )

    # Compact prompt — keep tokens low to avoid free-tier quota hits
    movie_list = []
    for m in movies:
        movie_list.append({
            "id":       m["id"],
            "title":    m["title"],
            "genres":   m["genres"],
            "rating":   m["rating"],
            "overview": m["overview"][:200],
            "language": m["language"],
        })

    prompt = (
        f"You are a movie recommendation agent. Score each movie for this user.\n\n"
        f"USER TASTE: {profile_summary}\n\n"
        f"MOVIES:\n{json.dumps(movie_list, ensure_ascii=False)}\n\n"
        f'Return ONLY a JSON array, no markdown:\n'
        f'[{{"id":<id>,"match_score":<0-100>,"match_reason":"<max 12 words>"}}]'
    )

    # Retry up to 3 times on quota errors with backoff
    raw = None
    for attempt in range(2):
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
        log.warning("Gemini unavailable after retries. Falling back to genre-based scoring.")
        loved = set(g.lower() for g in TASTE_PROFILE["loved_genres"])
        enriched = []
        for m in movies:
            overlap = len(loved & set(g.lower() for g in m.get("genres", [])))
            score = min(50 + overlap * 20, 95)
            enriched.append({**m, "match_score": score, "match_reason": "Scored by genre match (AI unavailable)."})
        enriched.sort(key=lambda x: x["match_score"], reverse=True)
        return enriched

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores = json.loads(raw)
    score_map = {s["id"]: s for s in scores}

    # Merge scores back into movie dicts
    enriched = []
    for m in movies:
        s = score_map.get(m["id"], {"match_score": 0, "match_reason": "No match data."})
        enriched.append({**m, **s})

    enriched.sort(key=lambda x: x["match_score"], reverse=True)
    log.info("AI scoring complete. Top movie: %s (score %s)",
             enriched[0]["title"] if enriched else "none",
             enriched[0]["match_score"] if enriched else 0)
    return enriched


# ---------------------------------------------------------------------------
# Email builder
# ---------------------------------------------------------------------------

LANG_MAP = {"en": "English", "te": "Telugu", "hi": "Hindi"}

def _runtime_str(minutes: int) -> str:
    if not minutes:
        return ""
    h, m = divmod(minutes, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _platform_badges(platforms: list[str]) -> str:
    colors = {
        "Netflix":      ("#E50914", "#fff"),
        "Prime Video":  ("#00A8E1", "#fff"),
        "Disney+ Hotstar": ("#1D6EC5", "#fff"),
        "Zee5":         ("#8B2FC9", "#fff"),
        "SonyLIV":      ("#D2182B", "#fff"),
    }
    badges = ""
    for p in platforms[:3]:
        bg, fg = colors.get(p, ("#444", "#fff"))
        badges += (
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:99px;font-size:11px;margin-right:4px;">{p}</span>'
        )
    return badges or '<span style="color:#888;font-size:12px;">OTT platform TBC</span>'


def build_email_html(movies: list[dict]) -> str:
    today_str = datetime.utcnow().strftime("%B %d, %Y")
    top = movies[0] if movies else None

    cards = ""
    for i, m in enumerate(movies):
        border = "border: 1.5px solid #185FA5;" if i == 0 else "border: 1px solid #e0e0e0;"
        top_badge = (
            '<span style="background:#E6F1FB;color:#0C447C;font-size:11px;'
            'padding:2px 8px;border-radius:99px;margin-bottom:8px;display:inline-block;">'
            '⭐ Top pick for you</span><br>' if i == 0 else ""
        )
        genre_tags = "".join(
            f'<span style="background:#EEEDFE;color:#3C3489;font-size:11px;'
            f'padding:2px 8px;border-radius:99px;margin-right:4px;">{g}</span>'
            for g in m["genres"][:3]
        )
        lang_label = LANG_MAP.get(m["language"], m["language"].upper())
        runtime    = _runtime_str(m.get("runtime", 0))
        bar_width  = m["match_score"]
        bar_color  = "#185FA5" if bar_width >= 70 else "#639922" if bar_width >= 50 else "#BA7517"

        cards += f"""
<div style="background:#fff;{border}border-radius:12px;padding:16px;margin-bottom:12px;">
  {top_badge}
  <div style="font-size:17px;font-weight:500;color:#1a1a1a;margin-bottom:4px;">{m['title']}</div>
  <div style="font-size:12px;color:#666;margin-bottom:8px;">
    ⭐ {m['rating']} &nbsp;·&nbsp; {lang_label} &nbsp;·&nbsp; {runtime}
  </div>
  <div style="margin-bottom:8px;">{genre_tags}</div>
  <div style="margin-bottom:8px;">{_platform_badges(m.get('platforms', []))}</div>
  <div style="margin-bottom:6px;">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#666;margin-bottom:3px;">
      <span>Taste match</span><span style="font-weight:500;color:#1a1a1a;">{m['match_score']}%</span>
    </div>
    <div style="height:4px;background:#eee;border-radius:99px;">
      <div style="width:{bar_width}%;height:4px;background:{bar_color};border-radius:99px;"></div>
    </div>
  </div>
  <div style="font-size:12px;color:#666;font-style:italic;">{m.get('match_reason','')}</div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;margin:0;padding:24px;">
<div style="max-width:560px;margin:0 auto;">

  <!-- Header -->
  <div style="background:#185FA5;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <div style="font-size:22px;font-weight:500;color:#fff;margin-bottom:4px;">🎬 Your Friday Movie Picks</div>
    <div style="font-size:13px;color:#B5D4F4;">{today_str} &nbsp;·&nbsp; Curated by your Movie Agent</div>
  </div>

  <!-- Intro -->
  <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 20px;margin-bottom:16px;font-size:14px;color:#444;line-height:1.6;">
    Hey! I scanned all new OTT releases in India this week across Netflix, Prime Video, Hotstar and more.
    Here are your top <strong>{len(movies)} picks</strong> matched to your taste —
    <strong>Comedy, Romance & Thriller</strong> in English, Telugu and Hindi.
  </div>

  <!-- Movie cards -->
  {cards}

  <!-- Footer -->
  <div style="font-size:11px;color:#aaa;text-align:center;margin-top:16px;line-height:1.6;">
    Powered by your personal Movie Agent · TMDB + Gemini AI<br>
    Runs every Friday at 7 PM IST via GitHub Actions · Completely free
  </div>

</div>
</body>
</html>"""
    return html


def tool_send_email(movies: list[dict]) -> bool:
    """
    Tool: Compose and send the weekly digest email via Gmail SMTP.
    """
    log.info("Tool called: send_email")

    sender    = os.environ["GMAIL_ADDRESS"].strip()
    recipient = os.environ["RECIPIENT_EMAIL"].strip()
    password  = "".join(c for c in os.environ["GMAIL_APP_PASSWORD"] if c.isascii() and not c.isspace())

    log.info("Sender:          %s", sender)
    log.info("Recipient:       %s", recipient)
    log.info("Password length: %d chars (should be 16)", len(password))

    html = build_email_html(movies)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎬 Your Friday movie picks — {len(movies)} new releases matched to you"
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html"))

    try:
        log.info("Connecting to smtp.gmail.com:465 ...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.set_debuglevel(1)
            log.info("Logging in ...")
            server.login(sender, password)
            log.info("Login successful. Sending message ...")
            server.sendmail(sender, recipient, msg.as_string())
            log.info("✅ Email sent successfully to %s", recipient)
        return True
    except smtplib.SMTPAuthenticationError as e:
        log.error("❌ Gmail authentication failed: %s", e)
        log.error("Fix: Go to myaccount.google.com/apppasswords, delete the old password,")
        log.error("     generate a new one, and update the GMAIL_APP_PASSWORD GitHub secret.")
        raise
    except smtplib.SMTPException as e:
        log.error("❌ SMTP error while sending: %s", e)
        raise
    except Exception as e:
        log.error("❌ Unexpected error: %s", e)
        raise


# ---------------------------------------------------------------------------
# The Agent — reasoning loop
# ---------------------------------------------------------------------------

def run_agent():
    """
    Agentic reasoning loop.

    The agent is given a goal and decides which tools to call,
    in what order, and when it has enough information to finish.
    """
    log.info("=" * 60)
    log.info("Movie Recommendation Agent starting")
    log.info("Goal: Find the best new OTT movie releases for the user and send a digest email.")
    log.info("=" * 60)

    # --- Step 1: Agent decides to search for new releases ---
    log.info("Agent reasoning: I need to find what's new on OTT this week.")
    raw_movies = tool_search_new_releases()

    if not raw_movies:
        log.warning("Agent: No movies found at all. Aborting.")
        return

    # --- Step 2: Agent decides to filter by language & rating ---
    log.info("Agent reasoning: %d movies found. Too many — I'll filter by user's preferred languages and minimum rating.", len(raw_movies))
    filtered = tool_filter_by_language(raw_movies)

    if not filtered:
        log.warning("Agent: No movies left after filtering. Trying with a lower rating bar.")
        # Agent adapts: retry with a lower threshold
        original_min = TASTE_PROFILE["min_rating"]
        TASTE_PROFILE["min_rating"] = 4.0
        filtered = tool_filter_by_language(raw_movies)
        TASTE_PROFILE["min_rating"] = original_min
        if not filtered:
            log.error("Agent: Still no movies. Giving up.")
            return

    # --- Step 3: Agent decides to fetch full details for each movie ---
    log.info("Agent reasoning: I have %d candidates. I need ratings, genres and platforms before I can score them.", len(filtered))
    detailed = []
    for m in filtered:
        details = tool_get_movie_details(m["id"])
        if details:
            detailed.append(details)

    if not detailed:
        log.error("Agent: Could not fetch details for any movie. Aborting.")
        return

    # Agent filters out hated genres before calling AI (saves tokens)
    hated = set(g.lower() for g in TASTE_PROFILE["hated_genres"])
    pre_filtered = [
        m for m in detailed
        if not any(g.lower() in hated for g in m.get("genres", []))
    ]
    log.info("Agent reasoning: After removing hated genres, %d movies remain. Sending to AI scorer.", len(pre_filtered))

    if not pre_filtered:
        log.warning("Agent: All remaining movies are in hated genres. Nothing to recommend.")
        return

    # --- Step 4: Agent calls Gemini to score each movie ---
    log.info("Agent reasoning: Calling Gemini to score each movie against the user's taste profile.")
    scored = tool_score_taste_match(pre_filtered)

    # Agent decides: keep only top N with score above 40
    top_picks = [m for m in scored if m["match_score"] >= 40][: TASTE_PROFILE["top_n"]]

    if not top_picks:
        log.info("Agent reasoning: No movies scored above 40. Taking top 3 anyway.")
        top_picks = scored[: 3]

    log.info("Agent reasoning: I have %d high-quality picks. Composing and sending the email.", len(top_picks))

    # --- Step 5: Agent sends the email ---
    success = tool_send_email(top_picks)

    if success:
        log.info("Agent: Goal achieved. Email delivered. Shutting down.")
    else:
        log.error("Agent: Email delivery failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Validate required env vars before starting
    required = ["TMDB_API_KEY", "GEMINI_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    run_agent()