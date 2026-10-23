# JOSE - Autonomous Execution & Intelligence System

JOSE turns a plain-language objective into a plan of tasks, runs them with a set of agents and tools
(browser, web search, files, Excel/PDF/PowerPoint, email, Telegram, ...), verifies the results and
recovers from failures. Risky actions need your approval, and every run is recorded in a ledger.

It ships as a FastAPI backend and a terminal interface (CLI) that share the same engine.

## Features

- Objective -> intent analysis -> plan -> approval -> execution -> verification -> recovery
- Multiple agents: research, browser, workspace, messenger, exam, verification, recovery, custom workflows
- Risk-based approval for sensitive actions, plus a sandboxed filesystem
- LLM via Gemini, with automatic Groq fallback; runs in demo mode with no API key
- Report and artifact generation for every run

## Requirements

- Python 3.11+
- (Optional) Playwright Chromium for browser tasks

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env            # then edit .env
```

Leave `GEMINI_API_KEY` empty to run in demo mode. See `.env.example` for all options
(search provider, SMTP, Telegram, Groq fallback, filesystem sandbox, ...).

**Never commit your `.env` file** - it is git-ignored.

## Usage

Terminal (from `backend/`):

```bash
python -m app.cli                              # voice mode, falls back to typed prompt
python -m app.cli repl                         # typed prompt
python -m app.cli run "your objective" [-f file ...] [--auto] [--simulate] [-y]
python -m app.cli history | show RUN | graph RUN | report RUN | artifacts RUN | continue RUN | cancel RUN
python -m app.cli tools
```

On Windows you can also use `jose.cmd` instead of `python -m app.cli`.

API server:

```bash
uvicorn app.main:app --reload
```

## Tests

```bash
cd backend
pytest
```

## Project layout

```
backend/
  app/
    agents/     planner, orchestrator and specialised agents
    api/        HTTP routes
    models/     state machine and DB tables
    services/   execution, approvals, ledger, LLM, reports
    tools/      browser, files, Excel/PDF/PPT, email, Telegram, search
    cli.py      terminal interface
  tests/
  data/         runtime data (git-ignored)
```
