"""
Movie & Series Recommendation Agent — with OTT filter, feedback memory, multi-recipient
-----------------------------------------------------------------------------------------
Requirements:
    pip install requests google-genai

GitHub Secrets needed:
    TMDB_API_KEY        — themoviedb.org (free)
    GEMINI_API_KEY      — aistudio.google.com (free)
    GMAIL_ADDRESS       — sender Gmail
    GMAIL_APP_PASSWORD  — Gmail App Password (16 chars, no spaces)
    RECIPIENT_EMAILS    — comma-separated: you@gmail.com,friend@gmail.com
    GITHUB_TOKEN        — auto-provided by GitHub Actions (for writing feedback.json)
    GITHUB_REPO         — your repo e.g. username/movie-agent
"""

import os, json, time, smtplib, logging, base64, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from google import genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taste profile (static baseline — feedback memory enriches this at runtime)
# ---------------------------------------------------------------------------
TASTE_PROFILE = {
    "loved_genres":   ["Comedy", "Romance", "Thriller"],
    "hated_genres":   ["Horror", "Documentary"],
    "languages":      ["en", "te", "hi"],
    "language_names": ["English", "Telugu", "Hindi"],
    "min_rating":     5.0,
    "top_n":          6,
}
LANG_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi"}
TMDB_BASE  = "https://api.themoviedb.org/3"
FEEDBACK_FILE = "feedback.json"

# ---------------------------------------------------------------------------
# TMDB helper
# ---------------------------------------------------------------------------
def tmdb_get(endpoint, params):
    params["api_key"] = os.environ["TMDB_API_KEY"]
    r = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Tool 1 — Search new movies
# ---------------------------------------------------------------------------
def tool_search_new_movies():
    log.info("Tool: search_new_movies")
    today     = datetime.utcnow().date()
    days_ago  = today - timedelta(days=14)
    all_items = []
    for lang in TASTE_PROFILE["languages"]:
        for page in [1, 2]:
            data   = tmdb_get("/discover/movie", {
                "with_original_language": lang,
                "primary_release_date.gte": str(days_ago),
                "primary_release_date.lte": str(today),
                "sort_by": "release_date.desc",
                "vote_count.gte": 3,
                "page": page,
            })
            items = data.get("results", [])
            if not items: break
            all_items.extend(items)
    # trending top-up
    for m in tmdb_get("/trending/movie/week", {}).get("results", []):
        if m.get("original_language") in TASTE_PROFILE["languages"]:
            all_items.append(m)
    return _dedup(all_items, "movie")

# ---------------------------------------------------------------------------
# Tool 2 — Search new series
# ---------------------------------------------------------------------------
def tool_search_new_series():
    log.info("Tool: search_new_series")
    today    = datetime.utcnow().date()
    days_ago = today - timedelta(days=14)
    all_items = []
    for lang in TASTE_PROFILE["languages"]:
        for page in [1, 2]:
            data  = tmdb_get("/discover/tv", {
                "with_original_language": lang,
                "first_air_date.gte": str(days_ago),
                "first_air_date.lte": str(today),
                "sort_by": "first_air_date.desc",
                "vote_count.gte": 3,
                "page": page,
            })
            items = data.get("results", [])
            if not items: break
            all_items.extend(items)
    for m in tmdb_get("/trending/tv/week", {}).get("results", []):
        if m.get("original_language") in TASTE_PROFILE["languages"]:
            all_items.append(m)
    return _dedup(all_items, "tv")

def _dedup(items, media_type):
    seen, unique = set(), []
    for m in items:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            m["media_type"] = media_type
            unique.append(m)
    counts = Counter(m.get("original_language","?") for m in unique)
    log.info("  %s deduped: %d | %s", media_type, len(unique), dict(counts))
    return unique

# ---------------------------------------------------------------------------
# Tool 3 — Get full details + verify OTT availability in India
# ---------------------------------------------------------------------------
def tool_get_details_and_verify_ott(item_id, media_type):
    try:
        ep      = f"/{'movie' if media_type=='movie' else 'tv'}/{item_id}"
        details = tmdb_get(ep, {"append_to_response": "watch/providers", "language": "en-US"})

        # --- OTT verification: must be available in India ---
        providers_raw = details.get("watch/providers", {}).get("results", {}).get("IN", {})
        flatrate = providers_raw.get("flatrate", [])
        free_p   = providers_raw.get("free", [])
        rent     = providers_raw.get("rent", [])
        all_providers = flatrate + free_p
        platforms = list({p["provider_name"] for p in all_providers})

        # Strict OTT check: must have at least one streaming provider in India
        if not platforms:
            # Allow rent-only if it's a known major platform (some Indian platforms show as rent)
            rent_names = {p["provider_name"] for p in rent}
            major = {"Netflix","Amazon Prime Video","Prime Video","Disney+ Hotstar",
                     "Zee5","SonyLIV","JioCinema","Apple TV+","Mubi","Aha"}
            if not (rent_names & major):
                log.info("  Skipping %d — not on any Indian OTT", item_id)
                return None
            platforms = list(rent_names & major)

        genres   = [g["name"] for g in details.get("genres", [])]
        if media_type == "movie":
            title   = details.get("title", "Unknown")
            runtime = details.get("runtime", 0)
            release = details.get("release_date", "")
            seasons = episodes = None
        else:
            title    = details.get("name", "Unknown")
            runtime  = 0
            release  = details.get("first_air_date", "")
            seasons  = details.get("number_of_seasons", 1)
            episodes = details.get("number_of_episodes", 1)

        return {
            "id":         item_id,
            "media_type": media_type,
            "title":      title,
            "language":   details.get("original_language", ""),
            "overview":   details.get("overview", ""),
            "genres":     genres,
            "rating":     round(details.get("vote_average", 0), 1),
            "vote_count": details.get("vote_count", 0),
            "runtime":    runtime,
            "release":    release,
            "platforms":  platforms,
            "seasons":    seasons,
            "episodes":   episodes,
        }
    except Exception as exc:
        log.warning("  Details failed for %d: %s", item_id, exc)
        return None

# ---------------------------------------------------------------------------
# Tool 4 — Filter by language, rating, hated genres
# ---------------------------------------------------------------------------
def tool_filter(items):
    allowed = set(TASTE_PROFILE["languages"])
    hated   = {g.lower() for g in TASTE_PROFILE["hated_genres"]}
    out = []
    for m in items:
        if m.get("original_language") not in allowed: continue
        if (m.get("vote_average") or 0) < TASTE_PROFILE["min_rating"]: continue
        out.append(m)
    log.info("  Filter: %d → %d items", len(items), len(out))
    return out

# ---------------------------------------------------------------------------
# Tool 5 — Read feedback memory from GitHub
# ---------------------------------------------------------------------------
def tool_read_feedback():
    """
    Reads feedback.json from the GitHub repo.
    Returns dict: { movie_id: { title, thumbs, genres, language, timestamps } }
    """
    log.info("Tool: read_feedback_memory")
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        log.warning("  No GITHUB_TOKEN/REPO set — skipping feedback memory")
        return {}
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{FEEDBACK_FILE}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data    = json.loads(resp.read())
            content = base64.b64decode(data["content"]).decode("utf-8")
            feedback = json.loads(content)
            log.info("  Loaded feedback: %d entries", len(feedback))
            return feedback
    except Exception as e:
        log.warning("  feedback.json not found or unreadable (%s) — starting fresh", e)
        return {}

# ---------------------------------------------------------------------------
# Tool 6 — Build feedback-enriched taste summary for Gemini
# ---------------------------------------------------------------------------
def tool_build_taste_context(feedback: dict) -> str:
    base = (
        f"Loves: {', '.join(TASTE_PROFILE['loved_genres'])}. "
        f"Hates: {', '.join(TASTE_PROFILE['hated_genres'])}. "
        f"Languages: {', '.join(TASTE_PROFILE['language_names'])}."
    )
    if not feedback:
        return base

    liked    = [v for v in feedback.values() if v.get("thumbs") == "up"]
    disliked = [v for v in feedback.values() if v.get("thumbs") == "down"]

    # Count genre preferences from feedback
    liked_genres    = Counter(g for m in liked    for g in m.get("genres", []))
    disliked_genres = Counter(g for m in disliked for g in m.get("genres", []))

    context = base
    if liked:
        titles = [m["title"] for m in liked[-5:]]
        top_g  = [g for g,_ in liked_genres.most_common(3)]
        context += f" Previously liked: {', '.join(titles)}. Trending liked genres: {', '.join(top_g)}."
    if disliked:
        titles = [m["title"] for m in disliked[-5:]]
        top_g  = [g for g,_ in disliked_genres.most_common(3)]
        context += f" Previously disliked: {', '.join(titles)}. Avoid genres: {', '.join(top_g)}."

    log.info("  Taste context built from %d liked / %d disliked", len(liked), len(disliked))
    return context

# ---------------------------------------------------------------------------
# Tool 7 — Gemini AI scoring
# ---------------------------------------------------------------------------
def tool_score_taste_match(items, taste_context):
    log.info("Tool: score_taste_match (%d items)", len(items))
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    compact = [{
        "id":       m["id"],
        "type":     m["media_type"],
        "title":    m["title"],
        "genres":   m["genres"],
        "rating":   m.get("rating", 0),
        "overview": m.get("overview","")[:200],
        "language": m["language"],
    } for m in items]

    prompt = (
        f"You are a movie/series recommendation agent.\n\n"
        f"USER TASTE (including past feedback):\n{taste_context}\n\n"
        f"CONTENT TO SCORE:\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        f"Score each item 0-100. Items similar to previously liked = higher score. "
        f"Items similar to previously disliked = lower score.\n"
        f"Return ONLY a JSON array, no markdown:\n"
        f'[{{"id":<id>,"match_score":<0-100>,"match_reason":"<max 12 words>"}}]'
    )

    time.sleep(5)
    raw = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            raw  = resp.text.strip()
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                wait = 90 * (2 ** attempt)
                log.warning("  Gemini quota hit (%d/3). Waiting %ds...", attempt+1, wait)
                time.sleep(wait)
            else:
                raise

    if raw is None:
        log.warning("  Gemini failed — genre fallback scoring")
        loved = {g.lower() for g in TASTE_PROFILE["loved_genres"]}
        for m in items:
            overlap = len(loved & {g.lower() for g in m.get("genres",[])})
            m["match_score"]  = min(50 + overlap * 20, 95)
            m["match_reason"] = "Genre-based score (AI unavailable)."
        items.sort(key=lambda x: x["match_score"], reverse=True)
        return items

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    raw = raw.strip()

    scores    = json.loads(raw)
    score_map = {s["id"]: s for s in scores}
    enriched  = [{**m, **score_map.get(m["id"], {"match_score":0,"match_reason":""})} for m in items]
    enriched.sort(key=lambda x: x["match_score"], reverse=True)
    return enriched

# ---------------------------------------------------------------------------
# Tool 8 — Balanced picker (guarantee TE/HI/EN and movies+series)
# ---------------------------------------------------------------------------
def tool_pick_balanced(items, n):
    by_lang = defaultdict(list)
    for m in items:
        by_lang[m.get("language","")].append(m)

    picks, picked_ids = [], set()
    for lang in ["te", "hi", "en"]:
        if by_lang[lang]:
            top = by_lang[lang][0]
            picks.append(top)
            picked_ids.add(top["id"])

    for m in items:
        if len(picks) >= n: break
        if m["id"] not in picked_ids:
            picks.append(m)
            picked_ids.add(m["id"])

    log.info("Balanced picks: %s", [(m["title"], m.get("language"), m["media_type"]) for m in picks])
    return picks[:n]

# ---------------------------------------------------------------------------
# Feedback URL builder
# ---------------------------------------------------------------------------
def _feedback_url(movie_id, thumb, title, genres, language):
    base = os.environ.get("FEEDBACK_BASE_URL", "")
    if not base:
        return "#"
    params = urllib.parse.urlencode({
        "id":       movie_id,
        "thumb":    thumb,
        "title":    title[:60],
        "genres":   ",".join(genres[:3]),
        "lang":     language,
    })
    return f"{base}?{params}"

# ---------------------------------------------------------------------------
# Email builder
# ---------------------------------------------------------------------------
def _runtime_str(m):
    if m["media_type"] == "tv":
        s = m.get("seasons") or 1
        e = m.get("episodes") or 1
        return f"{s} season{'s' if s>1 else ''} · {e} ep"
    mins = m.get("runtime") or 0
    if not mins: return ""
    h, mi = divmod(mins, 60)
    return f"{h}h {mi}m" if h else f"{mi}m"

def _platform_badges(platforms):
    colors = {
        "Netflix":          ("#E50914","#fff"),
        "Amazon Prime Video":("#00A8E1","#fff"),
        "Prime Video":      ("#00A8E1","#fff"),
        "Disney+ Hotstar":  ("#1D6EC5","#fff"),
        "Zee5":             ("#8B2FC9","#fff"),
        "SonyLIV":          ("#D2182B","#fff"),
        "JioCinema":        ("#5C3BC0","#fff"),
        "Apple TV+":        ("#000","#fff"),
        "Aha":              ("#F5A623","#000"),
        "Mubi":             ("#000","#fff"),
    }
    out = ""
    for p in platforms[:3]:
        bg, fg = colors.get(p, ("#555","#fff"))
        out += (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
                f'border-radius:99px;font-size:11px;margin-right:4px;">{p}</span>')
    return out or '<span style="color:#888;font-size:12px;">Platform TBC</span>'

def build_email_html(items, feedback):
    import urllib.parse
    today_str   = datetime.utcnow().strftime("%B %d, %Y")
    sender_addr = os.environ.get("GMAIL_ADDRESS", "movieagent@gmail.com")
    cards       = ""

    for i, m in enumerate(items):
        border    = "border:1.5px solid #185FA5;" if i == 0 else "border:1px solid #e0e0e0;"
        top_badge = (
            '<span style="background:#E6F1FB;color:#0C447C;font-size:11px;' +
            'padding:2px 8px;border-radius:99px;margin-bottom:8px;display:inline-block;">' +
            '⭐ Top pick</span><br>'
        ) if i == 0 else ""

        type_badge = (
            '<span style="background:#E1F5EE;color:#085041;font-size:11px;' +
            'padding:2px 8px;border-radius:99px;margin-right:4px;">Series</span>'
            if m["media_type"] == "tv" else
            '<span style="background:#EEEDFE;color:#3C3489;font-size:11px;' +
            'padding:2px 8px;border-radius:99px;margin-right:4px;">Movie</span>'
        )
        lang_label = LANG_NAMES.get(m.get("language", ""), m.get("language", "").upper())
        genre_tags = "".join(
            f'<span style="background:#FFF3E0;color:#E65100;font-size:11px;' +
            f'padding:2px 8px;border-radius:99px;margin-right:4px;">{g}</span>'
            for g in m.get("genres", [])[:3]
        )
        bar       = m.get("match_score", 0)
        bar_color = "#185FA5" if bar >= 70 else "#639922" if bar >= 50 else "#BA7517"

        cards += f"""
<div style="background:#fff;{border}border-radius:12px;padding:16px;margin-bottom:12px;">
  {top_badge}
  <div style="margin-bottom:6px;">{type_badge}
    <span style="background:#F5F5F5;color:#555;font-size:11px;padding:2px 8px;border-radius:99px;">{lang_label}</span>
  </div>
  <div style="font-size:17px;font-weight:500;color:#1a1a1a;margin-bottom:4px;">{m["title"]}</div>
  <div style="font-size:12px;color:#666;margin-bottom:8px;">⭐ {m.get("rating", 0)} &nbsp;·&nbsp; {_runtime_str(m)}</div>
  <div style="margin-bottom:8px;">{genre_tags}</div>
  <div style="margin-bottom:10px;">{_platform_badges(m.get("platforms", []))}</div>
  <div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#666;margin-bottom:3px;">
      <span>Taste match</span><span style="font-weight:500;color:#1a1a1a;">{bar}%</span>
    </div>
    <div style="height:4px;background:#eee;border-radius:99px;">
      <div style="width:{bar}%;height:4px;background:{bar_color};border-radius:99px;"></div>
    </div>
  </div>
  <div style="font-size:12px;color:#666;font-style:italic;margin-top:6px;">{m.get("match_reason", "")}</div>
</div>"""

    # ── Combined feedback section at the bottom ──────────────────────────────
    # Build one mailto link whose body lists ALL movies pre-filled
    # The user just edits the ratings (👍 / 👎 / skip) and sends ONE email.
    fb_lines_prefilled = []
    for idx, m in enumerate(items):
        genre_str = ", ".join(m.get("genres", [])[:3])
        fb_lines_prefilled.append(
            f"{idx+1}. {m['title']} | id:{m['id']} | {m.get('language','')} | {genre_str} | {m['media_type']} | RATING: 👍 or 👎"
        )


    footer = "\n\n(Do not edit below this line)\nweek:" + today_str
    body_text = (
        "Rate each movie. Replace the word RATING with thumbs-up or thumbs-down.\n"
        "Delete rows you want to skip. Do not change anything else.\n\n"
        + "\n".join(fb_lines_prefilled)
        + footer
    )
    mailto_body    = urllib.parse.quote(body_text)
    mailto_subject = urllib.parse.quote(f"My feedback — {today_str}")
    mailto_href    = f"mailto:{sender_addr}?subject={mailto_subject}&body={mailto_body}"

    # Build the visual row list for display in the email
    fb_rows = ""
    for idx, m in enumerate(items):
        lang_label = LANG_NAMES.get(m.get("language", ""), m.get("language", "").upper())
        platform   = m.get("platforms", [""])[0] if m.get("platforms") else ""
        fb_rows += f"""
<tr>
  <td style="padding:8px 0;border-bottom:0.5px solid #f0f0f0;font-size:13px;color:#1a1a1a;vertical-align:middle;">
    <span style="background:#F5F5F5;color:#888;font-size:11px;padding:1px 7px;border-radius:99px;margin-right:6px;">{idx+1}</span>
    {m["title"]}
    <span style="font-size:11px;color:#999;margin-left:4px;">{lang_label} · {platform}</span>
  </td>
</tr>"""

    feedback_section = f"""
<div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 20px;margin-top:8px;">
  <div style="font-size:14px;font-weight:500;color:#1a1a1a;margin-bottom:4px;">Rate this week's picks</div>
  <div style="font-size:12px;color:#666;margin-bottom:12px;line-height:1.6;">
    Click the button below — it opens one email with all picks pre-filled.
    Just replace <strong>👍 or 👎</strong> for each one and hit send. Takes 10 seconds.
  </div>
  <table style="width:100%;border-collapse:collapse;">{fb_rows}</table>
  <div style="margin-top:14px;">
    <a href="{mailto_href}"
       style="display:block;text-align:center;background:#185FA5;color:#fff;
              padding:10px 20px;border-radius:8px;text-decoration:none;
              font-size:14px;font-weight:500;">
      ✉️ Open feedback email (all {len(items)} picks)
    </a>
  </div>
  <div style="font-size:11px;color:#aaa;text-align:center;margin-top:8px;line-height:1.5;">
    One email · all ratings · agent learns your taste next Friday
  </div>
</div>"""

    movies_c = sum(1 for m in items if m["media_type"] == "movie")
    series_c = len(items) - movies_c

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;margin:0;padding:24px;">
<div style="max-width:560px;margin:0 auto;">
  <div style="background:#185FA5;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
    <div style="font-size:22px;font-weight:500;color:#fff;margin-bottom:4px;">🎬 Your Friday Picks</div>
    <div style="font-size:13px;color:#B5D4F4;">{today_str} · OTT releases · EN / TE / HI</div>
  </div>
  <div style="background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:16px 20px;margin-bottom:16px;font-size:14px;color:#444;line-height:1.6;">
    <strong>{movies_c} movie{"s" if movies_c != 1 else ""} + {series_c} series</strong> on Indian OTT this week,
    matched to your taste — Comedy · Romance · Thriller.
  </div>
  {cards}
  {feedback_section}
  <div style="font-size:11px;color:#aaa;text-align:center;margin-top:16px;line-height:1.6;">
    Powered by your Movie Agent · TMDB + Gemini AI · Runs every Friday 7 PM IST
  </div>
</div></body></html>"""


def tool_send_email(items, feedback):
    log.info("Tool: send_email")
    sender    = os.environ["GMAIL_ADDRESS"].strip()
    password  = "".join(c for c in os.environ["GMAIL_APP_PASSWORD"] if c.isascii() and not c.isspace())
    recipients = [r.strip() for r in os.environ["RECIPIENT_EMAILS"].split(",") if r.strip()]
    log.info("  Sending to %d recipient(s): %s", len(recipients), recipients)

    movies_c = sum(1 for m in items if m["media_type"]=="movie")
    series_c = len(items) - movies_c
    html     = build_email_html(items, feedback)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎬 Friday picks — {movies_c} movies + {series_c} series on OTT this week"
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        log.info("  ✅ Email sent to %s", recipients)
        return True
    except smtplib.SMTPAuthenticationError as e:
        log.error("  ❌ Gmail auth failed: %s", e)
        raise
    except Exception as e:
        log.error("  ❌ Email error: %s", e)
        raise

# ---------------------------------------------------------------------------
# Agent reasoning loop
# ---------------------------------------------------------------------------
def run_agent():
    log.info("="*60)
    log.info("Movie & Series Agent — OTT only | feedback memory | multi-recipient")
    log.info("="*60)

    # Step 1: Read feedback memory
    log.info("Agent: Reading feedback memory to enrich taste profile...")
    feedback = tool_read_feedback()

    # Step 2: Search movies + series
    log.info("Agent: Searching new releases (movies + series) in EN/TE/HI...")
    raw_movies = tool_search_new_movies()
    raw_series = tool_search_new_series()
    all_raw    = raw_movies + raw_series
    log.info("Agent: %d movies + %d series = %d total", len(raw_movies), len(raw_series), len(all_raw))

    # Step 3: Basic language/rating filter
    log.info("Agent: Filtering by language and minimum rating...")
    filtered = tool_filter(all_raw)
    if not filtered:
        TASTE_PROFILE["min_rating"] = 4.0
        filtered = tool_filter(all_raw)
    if not filtered:
        log.error("Agent: Nothing found. Aborting.")
        return

    # Step 4: Fetch full details AND verify OTT availability in India
    log.info("Agent: Fetching details and verifying OTT availability for %d candidates...", len(filtered))
    detailed, ott_skipped = [], 0
    for m in filtered:
        d = tool_get_details_and_verify_ott(m["id"], m.get("media_type","movie"))
        if d:
            detailed.append(d)
        else:
            ott_skipped += 1
    log.info("Agent: %d confirmed on Indian OTT, %d skipped (not streaming)", len(detailed), ott_skipped)

    # Remove hated genres
    hated = {g.lower() for g in TASTE_PROFILE["hated_genres"]}
    clean = [m for m in detailed if not any(g.lower() in hated for g in m.get("genres",[]))]
    if not clean:
        clean = detailed

    # Step 5: Build taste context from feedback + score with Gemini
    log.info("Agent: Building taste context from feedback history...")
    taste_context = tool_build_taste_context(feedback)

    log.info("Agent: Scoring %d OTT-confirmed items with Gemini...", len(clean))
    scored = tool_score_taste_match(clean, taste_context)

    # Step 6: Balanced pick
    good      = [m for m in scored if m.get("match_score",0) >= 35] or scored
    top_picks = tool_pick_balanced(good, TASTE_PROFILE["top_n"])

    # Step 7: Send to all recipients
    log.info("Agent: Sending email to all recipients...")
    tool_send_email(top_picks, feedback)
    log.info("Agent: Done! ✅")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    required = ["TMDB_API_KEY","GEMINI_API_KEY","GMAIL_ADDRESS","GMAIL_APP_PASSWORD","RECIPIENT_EMAILS","GITHUB_REPO"]
    missing  = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")
    run_agent()