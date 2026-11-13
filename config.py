"""
MeetMate - Agentic Google Calendar Assistant
config.py: Environment variable loading and app settings
"""
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/oauth/callback")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# App secret — MUST be overridden in production
_DEFAULT_SECRET = "dev-secret-key-change-in-prod"
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", _DEFAULT_SECRET)
if APP_SECRET_KEY == _DEFAULT_SECRET:
    warnings.warn(
        "APP_SECRET_KEY is using the insecure default. "
        "Set a strong random value in your .env file before deploying.",
        stacklevel=1,
    )

DEFAULT_TIMEZONE = "Asia/Kolkata"

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# Gemini model — override via GEMINI_MODEL env var
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TOOL_ROUNDS = 3

# Pending proposal TTL in seconds
PROPOSAL_TTL_SECONDS = 600  # 10 minutes
