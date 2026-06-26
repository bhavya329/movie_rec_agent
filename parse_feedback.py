"""
parse_feedback.py
-----------------
Reads the combined feedback email the user sent (one email, all ratings),
parses each movie rating, and saves to feedback.json in the GitHub repo.

Email format the agent sends:
    1. Fighter | id:12345 | hi | Thriller | movie | RATING: thumbs-up or thumbs-down
    2. Tillu Square | id:67890 | te | Comedy | tv | RATING: thumbs-up or thumbs-down

User edits it to:
    1. Fighter | id:12345 | hi | Thriller | movie | RATING: thumbs-up
    2. Tillu Square | id:67890 | te | Comedy | tv | RATING: thumbs-down

Requirements: pip install requests  (no extra packages needed)
"""

import os, json, imaplib, email, base64, urllib.request
from datetime import datetime

FEEDBACK_FILE = "feedback.json"

THUMBS_UP_WORDS   = {"thumbs-up", "👍", "up", "like", "liked", "yes", "good", "great"}
THUMBS_DOWN_WORDS = {"thumbs-down", "👎", "down", "dislike", "disliked", "no", "bad", "skip"}


def parse_rating(rating_str):
    """Convert user's free-text rating to 'up' or 'down'."""
    clean = rating_str.strip().lower()
    if any(w in clean for w in THUMBS_UP_WORDS):
        return "up"
    if any(w in clean for w in THUMBS_DOWN_WORDS):
        return "down"
    return None


def parse_feedback_body(body):
    """
    Parse combined feedback email body.
    Each rated line looks like:
        1. Fighter | id:12345 | hi | Thriller, Comedy | movie | RATING: thumbs-up
    Returns list of feedback dicts.
    """
    entries = []
    for line in body.splitlines():
        line = line.strip()
        # Skip empty lines and the footer
        if not line or line.startswith("(Do not edit") or line.startswith("week:"):
            continue
        # Must contain | and RATING:
        if "|" not in line or "RATING:" not in line.upper():
            continue
        # Strip leading number+dot
        if line[0].isdigit() and "." in line[:3]:
            line = line.split(".", 1)[1].strip()

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue

        title      = parts[0]
        id_part    = parts[1]  # "id:12345"
        language   = parts[2]
        genres_str = parts[3]
        media_type = parts[4]
        rating_raw = parts[5]  # "RATING: thumbs-up"

        movie_id = id_part.replace("id:", "").strip()
        genres   = [g.strip() for g in genres_str.split(",")]

        rating_value = rating_raw.split(":", 1)[-1].strip() if ":" in rating_raw else rating_raw
        thumb = parse_rating(rating_value)

        if thumb and movie_id:
            entries.append({
                "id":         movie_id,
                "title":      title,
                "thumbs":     thumb,
                "language":   language,
                "genres":     genres,
                "media_type": media_type,
                "timestamp":  datetime.utcnow().isoformat(),
            })
            print(f"  Parsed: {thumb} for '{title}'")

    return entries


def read_feedback_emails():
    """Scan Gmail inbox for feedback reply emails and parse them."""
    sender   = os.environ["GMAIL_ADDRESS"].strip()
    password = "".join(c for c in os.environ["GMAIL_APP_PASSWORD"] if c.isascii() and not c.isspace())

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(sender, password)
    mail.select("inbox")

    # Search for unread emails with "My feedback" in subject
    result, data = mail.search(None, '(UNSEEN SUBJECT "My feedback")')
    if result != "OK":
        print("  IMAP search failed")
        mail.logout()
        return []

    all_entries = []
    ids = data[0].split()
    print(f"  Found {len(ids)} unread feedback email(s)")

    for num in ids:
        result, msg_data = mail.fetch(num, "(RFC822)")
        if result != "OK":
            continue
        msg  = email.message_from_bytes(msg_data[0][1])
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        entries = parse_feedback_body(body)
        all_entries.extend(entries)

        # Mark as read
        mail.store(num, "+FLAGS", "\\Seen")
        print(f"  Email processed: {len(entries)} ratings found")

    mail.logout()
    return all_entries


def load_existing_feedback():
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        return {}, None
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{FEEDBACK_FILE}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data    = json.loads(resp.read())
            content = base64.b64decode(data["content"]).decode("utf-8")
            return json.loads(content), data["sha"]
    except Exception as e:
        print(f"  feedback.json not found — starting fresh ({e})")
        return {}, None


def save_feedback(feedback_dict, sha):
    token = os.environ["GITHUB_TOKEN"]
    repo  = os.environ["GITHUB_REPO"]
    url   = f"https://api.github.com/repos/{repo}/contents/{FEEDBACK_FILE}"
    payload = {
        "message": f"Update feedback [{datetime.utcnow().strftime('%Y-%m-%d')}]",
        "content": base64.b64encode(
            json.dumps(feedback_dict, indent=2, ensure_ascii=False).encode()
        ).decode(),
    }
    if sha:
        payload["sha"] = sha
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"  feedback.json saved to GitHub (HTTP {resp.status})")


if __name__ == "__main__":
    print("Scanning inbox for feedback emails...")
    new_entries = read_feedback_emails()

    if not new_entries:
        print("No new feedback found. Nothing to update.")
    else:
        print(f"Found {len(new_entries)} new rating(s). Updating feedback.json...")
        existing, sha = load_existing_feedback()
        for entry in new_entries:
            existing[entry["id"]] = entry
        save_feedback(existing, sha)
        print(f"Done. Total feedback entries: {len(existing)}")