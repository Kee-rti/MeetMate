"""
MeetMate - agent.py
Gemini-powered agent using the google-genai SDK with manual tool-call loop.

Key architectural change: create_calendar_event is NOT exposed as a Gemini tool.
The agent instead calls propose_meeting_creation (intercepted server-side) which
stores a pending proposal in the session. Actual calendar event creation only
happens when the user POSTs to /api/confirm with a valid proposal_id.
"""
import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOOL_ROUNDS, DEFAULT_TIMEZONE
from calendar_tools import (
    get_current_datetime,
    list_calendar_events,
    check_availability,
    get_event_details,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are MeetMate, an AI assistant that helps users schedule and manage Google Calendar events.

WORKFLOW (always follow this order):
1. Extract meeting details from the user's request.
2. Use get_current_datetime to know today's date/time.
3. Resolve any relative dates ("tomorrow", "next Monday") using the current date.
4. Use list_calendar_events or check_availability to verify the slot is free.
5. If the slot is busy, suggest the nearest available 30-minute slot on the same day.
6. When you have all required details AND the slot is confirmed available, call
   propose_meeting_creation with the full meeting details.
   The system will store a pending proposal and show the user a Confirm / Cancel UI.
7. DO NOT call propose_meeting_creation more than once per request.
8. DO NOT tell the user "I will create the event" — instead say "Here is your meeting
   preview — please confirm or cancel using the buttons below."
9. After the user confirms (via the UI), the backend creates and verifies the event
   automatically. You do not need to call any additional tool.

RULES:
- Never fabricate tool results.
- Ask for clarification only if the title, date, or time is genuinely missing.
- Use timezone from calendar if available, otherwise use {DEFAULT_TIMEZONE}.
- Maximum {MAX_TOOL_ROUNDS} tool-call rounds per user message.
- Format previews clearly using emoji: 📅 📋 ⏰ 👥.
- Attendees are optional — only include them if the user provides specific email addresses.
- Never claim an event was created unless the backend confirms it.
"""

# ── Tool declarations ────────────────────────────────────────────────────────

TOOL_DECLARATIONS = [
    genai_types.FunctionDeclaration(
        name="get_current_datetime",
        description=(
            "Get the current date, time and timezone. "
            "Call this first to resolve relative dates like 'tomorrow' or 'next Monday'."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "timezone_str": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="IANA timezone string, e.g. 'Asia/Kolkata'. Optional.",
                )
            },
        ),
    ),
    genai_types.FunctionDeclaration(
        name="list_calendar_events",
        description="List all calendar events on a specific date to check for conflicts.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "date_str": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Date in YYYY-MM-DD format.",
                ),
                "timezone_str": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="IANA timezone string. Optional.",
                ),
            },
            required=["date_str"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="check_availability",
        description="Check whether a specific time window is free on the primary calendar.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "start_datetime": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Start datetime in ISO 8601 format, e.g. '2024-01-15T15:00:00'.",
                ),
                "end_datetime": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="End datetime in ISO 8601 format.",
                ),
                "timezone_str": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="IANA timezone string. Optional.",
                ),
            },
            required=["start_datetime", "end_datetime"],
        ),
    ),
    genai_types.FunctionDeclaration(
        name="get_event_details",
        description="Retrieve details of a created event by its ID to verify creation.",
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "event_id": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Google Calendar event ID.",
                )
            },
            required=["event_id"],
        ),
    ),
    # propose_meeting_creation is intercepted server-side — the agent calls this
    # tool to surface a meeting preview; the backend stores it as a pending proposal.
    # The actual Google Calendar API call only happens after /api/confirm is POSTed.
    genai_types.FunctionDeclaration(
        name="propose_meeting_creation",
        description=(
            "Call this when you have verified availability and are ready to show the user "
            "a meeting preview for their confirmation. This does NOT create the event — "
            "it stores a pending proposal that the user must explicitly confirm via the UI."
        ),
        parameters=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "summary": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Event title.",
                ),
                "start_datetime": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Start datetime ISO 8601.",
                ),
                "end_datetime": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="End datetime ISO 8601.",
                ),
                "description": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Optional event description.",
                ),
                "attendees": genai_types.Schema(
                    type=genai_types.Type.ARRAY,
                    items=genai_types.Schema(type=genai_types.Type.STRING),
                    description="Optional attendee email addresses.",
                ),
                "timezone_str": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="IANA timezone string. Optional.",
                ),
            },
            required=["summary", "start_datetime", "end_datetime"],
        ),
    ),
]

GEMINI_TOOLS = [genai_types.Tool(function_declarations=TOOL_DECLARATIONS)]


class MeetMateAgent:
    def __init__(
        self,
        token_data: Optional[dict] = None,
        demo_mode: bool = False,
        proposal_callback=None,
    ):
        """
        proposal_callback: callable(proposal_dict) -> proposal_id
            Called when the LLM requests propose_meeting_creation.
            The callback stores the proposal server-side and returns its ID.
        """
        self.token_data = token_data
        self.demo_mode = demo_mode
        self.proposal_callback = proposal_callback
        self._init_gemini()

    def _init_gemini(self):
        if not GEMINI_API_KEY:
            self.client = None
            return
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _dispatch_tool(self, fn_name: str, fn_args: dict) -> dict:
        """Execute a tool call and return its result dict."""
        try:
            if fn_name == "get_current_datetime":
                return get_current_datetime(**fn_args)

            # propose_meeting_creation is intercepted here — no calendar API call
            if fn_name == "propose_meeting_creation":
                if self.proposal_callback:
                    proposal_id = self.proposal_callback(fn_args)
                    return {
                        "status": "proposal_stored",
                        "proposal_id": proposal_id,
                        "message": (
                            "Meeting preview has been prepared. "
                            "The user will see Confirm / Cancel buttons. "
                            "Do not claim the event is created yet."
                        ),
                    }
                return {"status": "proposal_stored", "proposal_id": "demo", "message": "Demo proposal stored."}

            # All other calendar tools require a connected account
            if not self.token_data and not self.demo_mode:
                return {"error": "Google Calendar not connected. Please connect your calendar first."}

            if self.demo_mode:
                return self._demo_tool(fn_name, fn_args)

            if fn_name == "list_calendar_events":
                return list_calendar_events(self.token_data, **fn_args)
            elif fn_name == "check_availability":
                return check_availability(self.token_data, **fn_args)
            elif fn_name == "get_event_details":
                return get_event_details(self.token_data, **fn_args)
            else:
                return {"error": f"Unknown tool: {fn_name}"}

        except Exception as e:
            logger.error("Tool %s error: %s", fn_name, e)
            return {"error": str(e)}

    def _demo_tool(self, fn_name: str, fn_args: dict) -> dict:
        """Demo mode: return plausible mock results without hitting the API."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(DEFAULT_TIMEZONE)
        now = datetime.now(tz)

        if fn_name == "list_calendar_events":
            return {
                "date": fn_args.get("date_str"),
                "timezone": DEFAULT_TIMEZONE,
                "events": [
                    {
                        "id": "demo_event_1",
                        "summary": "Team Standup",
                        "start": f"{fn_args.get('date_str')}T10:00:00+05:30",
                        "end": f"{fn_args.get('date_str')}T10:30:00+05:30",
                        "status": "confirmed",
                    }
                ],
            }
        elif fn_name == "check_availability":
            start = fn_args.get("start_datetime", "")
            if "T10:" in start:
                return {
                    "available": False,
                    "start": fn_args.get("start_datetime"),
                    "end": fn_args.get("end_datetime"),
                    "conflicts": [
                        {
                            "start": f"{now.date()}T10:00:00+05:30",
                            "end": f"{now.date()}T10:30:00+05:30",
                        }
                    ],
                }
            return {
                "available": True,
                "start": fn_args.get("start_datetime"),
                "end": fn_args.get("end_datetime"),
                "conflicts": [],
            }
        elif fn_name == "get_event_details":
            return {
                "event_id": fn_args.get("event_id"),
                "summary": "Demo Event",
                "start": now.isoformat(),
                "end": now.isoformat(),
                "status": "confirmed",
                "html_link": "https://calendar.google.com/calendar/event?eid=demo",
                "attendees": [],
                "description": "",
            }
        return {"error": f"Unknown demo tool: {fn_name}"}

    def chat(self, history: list[dict], user_message: str) -> tuple[str, list]:
        """
        Run the agent loop for one user message.

        Args:
            history: list of {"role": "user"|"model", "parts": [str]}
            user_message: the new user message text

        Returns:
            (assistant_reply, updated_history)
        """
        if not self.client:
            return (
                "⚠️ **Gemini API key not configured.** "
                "Please add your `GEMINI_API_KEY` to the `.env` file and restart the server. "
                "See `.env.example` for the required variable names.",
                history,
            )

        # Build contents list: history + new user turn
        contents: list[genai_types.Content] = []
        for msg in history:
            role = msg.get("role", "user")
            parts_data = msg.get("parts", [])
            parts = [genai_types.Part.from_text(text=p) if isinstance(p, str) else p for p in parts_data]
            contents.append(genai_types.Content(role=role, parts=parts))

        # Append current user message
        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_message)],
        ))

        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=GEMINI_TOOLS,
        )

        tool_rounds = 0
        response = None

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return f"❌ AI service error: {e}", history

        while tool_rounds < MAX_TOOL_ROUNDS:
            if not response.candidates:
                logger.warning("Gemini returned no candidates")
                break

            candidate = response.candidates[0]

            # Use the SDK's convenience accessor which filters parts with function_call set.
            # Falls back to manual scan if the attribute is absent (SDK version guard).
            fn_calls = getattr(response, "function_calls", None)
            if fn_calls is None:
                fn_calls = [
                    p.function_call
                    for p in candidate.content.parts
                    if p.function_call is not None
                ]

            if not fn_calls:
                break  # Text response — done

            tool_rounds += 1

            # Append model's tool-call turn to contents so it stays in context
            contents.append(candidate.content)

            # Execute tools and collect results.
            # Pass id= to from_function_response so the API can match each response
            # to its originating call — required for Gemini 2.5+ / 3.8 when the
            # model issues parallel calls or the same tool more than once per round.
            tool_response_parts = []
            for fc in fn_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                call_id = getattr(fc, "id", None)  # present in google-genai >= 1.x
                logger.info("Tool call: %s (id=%s)", fn_name, call_id)
                result = self._dispatch_tool(fn_name, fn_args)
                logger.debug("Tool result for %s: %s", fn_name, result)

                if call_id is None:
                    raise RuntimeError(
                        f"Gemini returned a function call without an ID: {fn_name}"
                    )
                tool_response_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id=call_id,
                            name=fn_name,
                            response={"result": result},
                        )
                    )
                )

            # generate_content only accepts "user" / "model" roles.
            # Function responses are sent back as role="user" content.
            contents.append(genai_types.Content(role="user", parts=tool_response_parts))

            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                logger.error("Gemini API error (tool round %d): %s", tool_rounds, e)
                return f"❌ AI service error after tool execution: {e}", history

        # Extract final text reply
        if response is None or not response.candidates:
            return "I couldn't generate a response. Please try again.", history

        text_parts = []
        for part in response.candidates[0].content.parts:
            if part.text:
                text_parts.append(part.text)

        reply = "\n".join(text_parts) if text_parts else "I couldn't generate a response. Please try again."

        # Build updated history (simple text format for round-trip to client)
        updated_history = list(history) + [
            {"role": "user", "parts": [user_message]},
            {"role": "model", "parts": [reply]},
        ]
        return reply, updated_history
