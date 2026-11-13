"""
MeetMate - main.py
FastAPI application: chat endpoint, OAuth flow, session management,
server-enforced event confirmation.
"""
import logging
import secrets
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature
from pydantic import BaseModel

from config import APP_SECRET_KEY, DEMO_MODE, GOOGLE_CLIENT_ID, PROPOSAL_TTL_SECONDS
from oauth_handler import get_authorization_url, exchange_code, load_token, revoke_token
from calendar_tools import create_calendar_event, get_event_details
from agent import MeetMateAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MeetMate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

signer = URLSafeTimedSerializer(APP_SECRET_KEY)

# ── Session store ────────────────────────────────────────────────────────────
# session_id -> {
#   "token_data": dict|None,
#   "oauth_state": str|None,          # pending OAuth state
#   "pending_proposals": {            # proposal_id -> proposal dict + expiry
#       "<id>": {"expires_at": float, "proposal": dict, "consumed": bool}
#   }
# }
sessions: dict[str, dict] = {}


# ── Session helpers ──────────────────────────────────────────────────────────

def _ensure_session(session: dict):
    """Ensure required keys exist in a session dict."""
    session.setdefault("token_data", None)
    session.setdefault("oauth_state", None)
    session.setdefault("oauth_code_verifier", None)
    session.setdefault("pending_proposals", {})


def _get_session(request: Request) -> dict:
    cookie = request.cookies.get("meetmate_session")
    if cookie:
        try:
            sid = signer.loads(cookie, max_age=86400 * 30)
            if sid in sessions:
                _ensure_session(sessions[sid])
                return sessions[sid]
        except BadSignature:
            pass
    # New session
    sid = secrets.token_hex(16)
    sessions[sid] = {}
    _ensure_session(sessions[sid])
    return sessions[sid]


def _get_session_id(request: Request) -> Optional[str]:
    cookie = request.cookies.get("meetmate_session")
    if cookie:
        try:
            return signer.loads(cookie, max_age=86400 * 30)
        except BadSignature:
            pass
    return None


def _set_session_cookie(response, sid: str):
    signed = signer.dumps(sid)
    response.set_cookie(
        "meetmate_session",
        signed,
        httponly=True,
        samesite="lax",
        max_age=86400 * 30,
    )


def _get_or_create_sid(request: Request) -> tuple[str, dict]:
    """Return (sid, session_dict), creating a new session if needed."""
    cookie = request.cookies.get("meetmate_session")
    if cookie:
        try:
            sid = signer.loads(cookie, max_age=86400 * 30)
            if sid in sessions:
                _ensure_session(sessions[sid])
                return sid, sessions[sid]
        except BadSignature:
            pass
    sid = secrets.token_hex(16)
    sessions[sid] = {}
    _ensure_session(sessions[sid])
    return sid, sessions[sid]


# ── Proposal helpers ─────────────────────────────────────────────────────────

def _store_proposal(session: dict, proposal: dict) -> str:
    """Store a pending meeting proposal. Returns an opaque proposal_id."""
    proposal_id = secrets.token_urlsafe(24)
    session["pending_proposals"][proposal_id] = {
        "expires_at": time.monotonic() + PROPOSAL_TTL_SECONDS,
        "proposal": proposal,
        "consumed": False,
    }
    return proposal_id


def _pop_proposal(session: dict, proposal_id: str) -> Optional[dict]:
    """
    Retrieve and consume a pending proposal if it is valid, unexpired,
    and not already used. Returns the proposal dict or None.
    """
    entry = session.get("pending_proposals", {}).get(proposal_id)
    if entry is None:
        return None
    if entry["consumed"]:
        logger.warning("Proposal %s already consumed", proposal_id)
        return None
    if time.monotonic() > entry["expires_at"]:
        logger.warning("Proposal %s expired", proposal_id)
        return None
    entry["consumed"] = True  # Mark consumed before creating event (idempotency)
    return entry["proposal"]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def status(request: Request):
    sid, session = _get_or_create_sid(request)
    token_data = session.get("token_data") or load_token()
    if token_data:
        session["token_data"] = token_data

    google_configured = bool(GOOGLE_CLIENT_ID)
    calendar_connected = token_data is not None

    auth_info = None
    if google_configured and not calendar_connected:
        result = get_authorization_url()
        if result:
            auth_url, state, code_verifier = result
            session["oauth_state"] = state
            session["oauth_code_verifier"] = code_verifier
            auth_info = auth_url

    response = JSONResponse({
        "calendar_connected": calendar_connected,
        "google_configured": google_configured,
        "demo_mode": DEMO_MODE,
        "auth_url": auth_info,
    })
    _set_session_cookie(response, sid)
    return response


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    sid, session = _get_or_create_sid(request)

    token_data = session.get("token_data") or load_token()
    if token_data:
        session["token_data"] = token_data

    # Build proposal callback so the agent can store proposals server-side
    def store_proposal(proposal: dict) -> str:
        return _store_proposal(session, proposal)

    agent = MeetMateAgent(
        token_data=token_data,
        demo_mode=DEMO_MODE,
        proposal_callback=store_proposal,
    )

    # Reconstruct history in simple format (agent.py handles conversion to genai types)
    history = []
    for msg in body.history:
        role = msg.get("role")
        text = msg.get("content", "")
        if role in ("user", "assistant") and text:
            history.append({
                "role": "user" if role == "user" else "model",
                "parts": [text],
            })

    reply, _ = agent.chat(history, body.message)

    # Check if a new proposal was stored during this turn
    proposals = session.get("pending_proposals", {})
    latest_proposal_id = None
    latest_proposal = None
    latest_time = 0.0
    for pid, entry in proposals.items():
        if not entry["consumed"] and entry["expires_at"] > latest_time:
            latest_time = entry["expires_at"]
            latest_proposal_id = pid
            latest_proposal = entry["proposal"]

    response_data: dict = {"reply": reply}
    if latest_proposal_id and latest_proposal:
        response_data["proposal"] = {
            "proposal_id": latest_proposal_id,
            "summary": latest_proposal.get("summary"),
            "start_datetime": latest_proposal.get("start_datetime"),
            "end_datetime": latest_proposal.get("end_datetime"),
            "description": latest_proposal.get("description", ""),
            "attendees": latest_proposal.get("attendees", []),
            "timezone_str": latest_proposal.get("timezone_str", "Asia/Kolkata"),
        }

    response = JSONResponse(response_data)
    _set_session_cookie(response, sid)
    return response


class ConfirmRequest(BaseModel):
    proposal_id: str


@app.post("/api/confirm")
async def confirm_event(request: Request, body: ConfirmRequest):
    """
    Server-enforced event creation.
    Only creates an event when:
      1. The proposal_id belongs to this session.
      2. The proposal exists, is unexpired, and has not been consumed.
    A model-generated statement or client boolean cannot substitute for this check.
    """
    sid, session = _get_or_create_sid(request)

    if not body.proposal_id:
        raise HTTPException(status_code=400, detail="proposal_id is required.")

    proposal = _pop_proposal(session, body.proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=400,
            detail="Proposal not found, already used, or expired. Please start a new request.",
        )

    token_data = session.get("token_data") or load_token()

    if DEMO_MODE:
        # Demo: return a fake success without hitting Google API
        response = JSONResponse({
            "success": True,
            "event": {
                "event_id": "demo_confirmed_xyz",
                "summary": proposal.get("summary"),
                "start": proposal.get("start_datetime"),
                "end": proposal.get("end_datetime"),
                "html_link": "https://calendar.google.com/calendar/event?eid=demo",
                "status": "confirmed",
            },
        })
        _set_session_cookie(response, sid)
        return response

    if not token_data:
        raise HTTPException(status_code=401, detail="Google Calendar not connected.")

    # Create the event
    created = create_calendar_event(
        token_data=token_data,
        summary=proposal["summary"],
        start_datetime=proposal["start_datetime"],
        end_datetime=proposal["end_datetime"],
        description=proposal.get("description"),
        attendees=proposal.get("attendees"),
        timezone_str=proposal.get("timezone_str"),
    )

    if "error" in created:
        raise HTTPException(status_code=502, detail=f"Calendar API error: {created['error']}")

    # Verify the event was actually created
    event_id = created.get("event_id")
    verified = None
    if event_id:
        verified = get_event_details(token_data=token_data, event_id=event_id)
        if "error" in verified:
            logger.warning("Event created but verification failed: %s", verified["error"])
            verified = None

    response = JSONResponse({
        "success": True,
        "event": verified or created,
    })
    _set_session_cookie(response, sid)
    return response


@app.post("/api/cancel")
async def cancel_proposal(request: Request, body: ConfirmRequest):
    """Discard a pending proposal."""
    sid, session = _get_or_create_sid(request)
    proposals = session.get("pending_proposals", {})
    entry = proposals.get(body.proposal_id)
    if entry:
        entry["consumed"] = True  # Mark consumed = effectively cancelled
    response = JSONResponse({"cancelled": True})
    _set_session_cookie(response, sid)
    return response


@app.get("/oauth/connect")
async def oauth_connect(request: Request):
    result = get_authorization_url()
    if not result:
        raise HTTPException(
            status_code=400,
            detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env",
        )
    auth_url, state, code_verifier = result
    sid, session = _get_or_create_sid(request)
    session["oauth_state"] = state
    session["oauth_code_verifier"] = code_verifier

    response = RedirectResponse(auth_url)
    _set_session_cookie(response, sid)
    return response


@app.get("/oauth/callback")
async def oauth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    if error:
        return RedirectResponse(f"/?error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code received.")
    if not state:
        raise HTTPException(status_code=400, detail="Missing OAuth state parameter.")

    sid, session = _get_or_create_sid(request)
    expected_state = session.get("oauth_state")
    expected_code_verifier = session.get("oauth_code_verifier")

    try:
        token_data = exchange_code(
            code=code,
            state=state,
            expected_state=expected_state,
            code_verifier=expected_code_verifier,
        )
    except ValueError as e:
        logger.warning("OAuth state validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("OAuth exchange failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to exchange authorization code.")

    if not token_data:
        raise HTTPException(status_code=500, detail="Failed to exchange authorization code.")

    # Consume the state
    session["oauth_state"] = None
    session["oauth_code_verifier"] = None
    session["token_data"] = token_data

    response = RedirectResponse("/?connected=true")
    _set_session_cookie(response, sid)
    return response


@app.post("/oauth/disconnect")
async def oauth_disconnect(request: Request):
    sid, session = _get_or_create_sid(request)
    session["token_data"] = None
    revoke_token()
    response = JSONResponse({"disconnected": True})
    _set_session_cookie(response, sid)
    return response


app.mount("/static", StaticFiles(directory="static"), name="static")
