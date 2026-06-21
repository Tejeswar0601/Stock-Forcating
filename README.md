# Forecast Terminal — Daily Auto-Updating Stock Predictions

A small, self-updating LSTM stock forecaster. Every day a GitHub Action retrains
the model on fresh data and writes the new 5-day forecast. The web page always
shows whatever the latest run produced — open it today, see today's 5 days;
open it tomorrow, see tomorrow's 5 days. No server, no manual steps once it's set up.

## What's in this repo

```
predict.py                     # training + prediction script (edit TICKERS list here)
requirements.txt               # Python deps for the Action runner
.github/workflows/daily.yml    # the daily schedule
index.html                     # the dashboard page (GitHub Pages serves this)
predictions/                   # JSON output, one file per ticker (auto-committed daily)
```

## Setup (one-time, ~10 minutes)

### 1. Create the GitHub repo
- Go to github.com → **New repository** → name it (e.g. `stock-forecast`) → Public → Create.
- Don't initialize with a README (you already have one here).

### 2. Push these files
From this folder, run:
```bash
git init
git add .
git commit -m "Initial commit: forecast terminal"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 3. Turn on GitHub Pages
- In your repo: **Settings → Pages**
- Under "Build and deployment" → Source: **Deploy from a branch**
- Branch: `main`, folder: `/ (root)` → **Save**
- After a minute, your live URL appears at the top of that page —
  something like `https://<your-username>.github.io/<repo-name>/`

### 4. Let the daily Action run
- The workflow in `.github/workflows/daily.yml` is already scheduled
  (weekdays at 23:30 UTC, after US markets close).
- To test it immediately instead of waiting: go to your repo's **Actions** tab →
  click **Daily Stock Forecast** → **Run workflow** → **Run workflow**.
- It takes a few minutes (training 3 LSTMs). When it finishes, check that
  `predictions/AAPL.json` etc. have new `generated_at_utc` timestamps.
- Refresh your GitHub Pages URL — the new forecast appears.

That's it. From here on, it updates itself every weekday with zero action from you.

## Changing the tracked tickers

Open `predict.py` and edit this line near the top:
```python
TICKERS = ["AAPL", "TSLA", "MSFT"]
```
Add/remove tickers (any valid Yahoo Finance symbol, e.g. `"INFY.NS"` for NSE-listed stocks),
commit, and push. The next scheduled run (or a manual run) will pick up the new list.

**Note on run time:** each ticker trains its own LSTM from scratch (~2-4 min on
GitHub's free runners). 3 tickers ≈ 10-12 minutes total, well within the free tier's
limits. Going much above 5-6 tickers may need a longer-running or self-hosted runner.

## Changing the schedule

Edit the `cron` line in `.github/workflows/daily.yml`. It uses standard cron syntax
in UTC. For example, to run once a day including weekends at 01:00 UTC:
```yaml
- cron: "0 1 * * *"
```

## How "always shows the next 5 days" works

`predict.py` always downloads data up through *today* and forecasts forward from
the most recent close — there's no hardcoded date. Each day's run naturally
produces "the next 5 trading days from whenever it ran." The web page has no
date logic of its own; it just displays whatever's in the latest JSON.

## Notes & honest limitations

- This is the same modeling approach as the original notebook: BiLSTM + Attention,
  trained on 15 technical indicators, predicting daily % returns rather than raw price.
- Markets are noisy. Treat `test_mae_return` (shown under the chart) as a rough
  confidence signal — daily return prediction for liquid stocks is inherently hard,
  and even a "good" model here is beating a coin flip, not predicting the future.
- This is for learning/portfolio purposes, not investment advice.
