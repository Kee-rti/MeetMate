"""
MeetMate - oauth_handler.py
Google OAuth 2.0 flow for single-user calendar access.
"""
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Optional

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_SCOPES,
    REDIRECT_URI,
)

logger = logging.getLogger(__name__)

TOKEN_FILE = Path("token.json")  # excluded from git
PENDING_OAUTH_FILE = Path("_pending_oauth.json")  # excluded from git


def _save_pending_state(state: str, code_verifier: str):
    """Store state and PKCE verifier so cross-origin redirects (e.g. 127.0.0.1 vs localhost) work."""
    data = {}
    if PENDING_OAUTH_FILE.exists():
        try:
            data = json.loads(PENDING_OAUTH_FILE.read_text())
        except Exception:
            data = {}
    data[state] = {"code_verifier": code_verifier, "created_at": time.time()}
    PENDING_OAUTH_FILE.write_text(json.dumps(data))


def get_and_consume_pending_verifier(state: str) -> Optional[str]:
    """Retrieve and consume PKCE verifier for a validated state, expiring after 10 minutes."""
    if not PENDING_OAUTH_FILE.exists():
        return None
    try:
        data = json.loads(PENDING_OAUTH_FILE.read_text())
        entry = data.pop(state, None)
        PENDING_OAUTH_FILE.write_text(json.dumps(data))
        if entry and isinstance(entry, dict):
            if time.time() - entry.get("created_at", 0) < 600:
                return entry.get("code_verifier")
            logger.warning("Pending OAuth state expired: %s", state)
    except Exception as e:
        logger.warning("Error reading pending oauth state: %s", e)
    return None


def build_flow(code_verifier: Optional[str] = None) -> Optional[Flow]:
    """Build OAuth flow, preserving PKCE verifier across requests."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return None
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=GOOGLE_SCOPES,
        redirect_uri=REDIRECT_URI,
        code_verifier=code_verifier,
        autogenerate_code_verifier=(code_verifier is None),
    )


def get_authorization_url() -> Optional[tuple[str, str, str]]:
    """
    Return auth_url, state and PKCE verifier.
    The caller must store state and verifier in the server session.
    """
    flow = build_flow()
    if not flow:
        return None
    state = secrets.token_urlsafe(32)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    _save_pending_state(state, flow.code_verifier)
    return auth_url, state, flow.code_verifier


def exchange_code(
    code: str,
    state: str,
    expected_state: Optional[str] = None,
    code_verifier: Optional[str] = None,
) -> Optional[dict]:
    """
    Validate OAuth state then exchange auth code for tokens.
    Raises ValueError on state mismatch.
    Persists tokens to token.json (git-ignored).
    Never logs secrets.
    """
    # Fallback if session cookie was lost across domains (e.g. 127.0.0.1 vs localhost)
    if not expected_state or not code_verifier:
        stored_verifier = get_and_consume_pending_verifier(state)
        if stored_verifier:
            expected_state = state
            code_verifier = stored_verifier

    if not state or not expected_state or not secrets.compare_digest(state, expected_state):
        raise ValueError("OAuth state mismatch or expired session — possible CSRF attack.")

    if not code_verifier:
        raise ValueError("Missing OAuth PKCE verifier. Restart Google login.")
    flow = build_flow(code_verifier=code_verifier)
    if not flow:
        return None
    flow.fetch_token(code=code)
    creds = flow.credentials
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else GOOGLE_SCOPES,
    }
    TOKEN_FILE.write_text(json.dumps(token_data))
    logger.info("OAuth tokens stored to %s (secrets not logged)", TOKEN_FILE)
    return token_data


def load_token() -> Optional[dict]:
    """Load and optionally refresh stored token. Returns token dict or None."""
    if not TOKEN_FILE.exists():
        return None
    try:
        token_data = json.loads(TOKEN_FILE.read_text())
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id", GOOGLE_CLIENT_ID),
            client_secret=token_data.get("client_secret", GOOGLE_CLIENT_SECRET),
            scopes=token_data.get("scopes", GOOGLE_SCOPES),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_data["token"] = creds.token
            TOKEN_FILE.write_text(json.dumps(token_data))
            logger.info("OAuth token refreshed (secret not logged)")
        return token_data
    except Exception as e:
        logger.warning("Token load/refresh failed: %s", e)
        return None


def revoke_token() -> bool:
    """Delete the stored token file."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    return True
