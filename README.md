# Movie Recommendation Agent

An agentic AI system that fetches new OTT releases every Friday,
scores them against your taste profile using Gemini, and sends a
curated digest email to your inbox — completely free.

## Stack

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
