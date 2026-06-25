# Movie Recommendation Agent

An agentic AI system that fetches new OTT releases every Friday,
scores them against your taste profile using Gemini, and sends a
curated digest email to your inbox — completely free.

## Stack

| Component | Service | Cost |
|---|---|---|
| Movie data | TMDB API | Free |
| AI scoring | Google Gemini 1.5 Flash | Free (1500 req/day) |
| Email | Gmail SMTP | Free |
| Scheduler | GitHub Actions | Free |

---

## Setup (15 minutes)

### 1. Get a TMDB API key
1. Sign up at https://www.themoviedb.org
2. Go to Settings → API → Request an API key
3. Copy your **API Key (v3 auth)**

### 2. Get a Gemini API key
1. Go to https://aistudio.google.com
2. Click **Get API key** → Create API key
3. Copy the key

### 3. Create a Gmail App Password
1. Enable 2-Step Verification on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Select app: **Mail** → Generate
4. Copy the 16-character password (no spaces)

### 4. Set up the GitHub repo
```bash
# Create a new repo and push these files
git init
git add agent.py .github/workflows/movie_agent.yml README.md
git commit -m "Add movie recommendation agent"
git remote add origin https://github.com/YOUR_USERNAME/movie-agent.git
git push -u origin main
```

### 5. Add secrets to GitHub
Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these 5 secrets:

| Secret name | Value |
|---|---|
| `TMDB_API_KEY` | Your TMDB v3 API key |
| `GEMINI_API_KEY` | Your Gemini API key |
| `GMAIL_ADDRESS` | your.email@gmail.com |
| `GMAIL_APP_PASSWORD` | The 16-char app password |
| `RECIPIENT_EMAIL` | Where to send the digest |

### 6. Test it manually
Go to your repo → **Actions** → **Movie Recommendation Agent** → **Run workflow**

Check your inbox within ~2 minutes!

---

## Customise your taste

Edit `agent.py`, find the `TASTE_PROFILE` dict near the top:

```python
TASTE_PROFILE = {
    "loved_genres":   ["Comedy", "Romance", "Thriller"],
    "hated_genres":   ["Horror", "Documentary"],
    "languages":      ["en", "te", "hi"],
    "min_rating":     5.5,   # minimum TMDB score (0-10)
    "top_n":          5,     # how many picks in the email
}
```

---

## How it works (agentic loop)

```
Friday 7 PM IST
       ↓
GitHub Actions wakes the agent
       ↓
Agent DECIDES → call search_new_releases()   (TMDB API)
       ↓
Agent REASONS → too many, filter by language/rating
       ↓
Agent DECIDES → call get_movie_details() for each
       ↓
Agent REASONS → drop hated genres before AI call
       ↓
Agent DECIDES → call score_taste_match()     (Gemini AI)
       ↓
Agent REASONS → keep scores ≥ 40, take top 5
       ↓
Agent DECIDES → call send_email()            (Gmail SMTP)
       ↓
Email in your inbox!
```

The AI (Gemini) reasons at every arrow — it is not a fixed script.