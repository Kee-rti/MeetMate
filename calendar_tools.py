"""
MeetMate - calendar_tools.py
Implements the 5 agent tools using Google Calendar API.
"""
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials

from config import DEFAULT_TIMEZONE


def _get_tz(tz_name: str) -> tuple:
    """
    Resolve a timezone name to a ZoneInfo object.
    On Windows, zoneinfo requires the 'tzdata' package.
    Falls back gracefully through DEFAULT_TIMEZONE → stdlib UTC.
    Returns (tzinfo_object, resolved_tz_name).
    """
    for name in (tz_name, DEFAULT_TIMEZONE):
        try:
            return ZoneInfo(name), name
        except (ZoneInfoNotFoundError, KeyError, ModuleNotFoundError):
            continue
    # Absolute fallback: stdlib UTC — always available, no packages needed
    return timezone.utc, "UTC"


def _get_service(token_data: dict):
    """Build a Google Calendar service from stored token dict."""
    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes"),
    )
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def get_current_datetime(timezone_str: Optional[str] = None) -> dict:
    """Tool: Return current date, time, and timezone."""
    tz, tz_name = _get_tz(timezone_str or DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "timezone": tz_name,
        "day_of_week": now.strftime("%A"),
    }


def list_calendar_events(token_data: dict, date_str: str, timezone_str: Optional[str] = None) -> dict:
    """Tool: List calendar events for a given date (YYYY-MM-DD)."""
    tz, tz_name = _get_tz(timezone_str or DEFAULT_TIMEZONE)
    try:
        day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
        day_end = day_start + timedelta(days=1)
        service = _get_service(token_data)
        result = service.events().list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = []
        for e in result.get("items", []):
            start = e.get("start", {})
            end = e.get("end", {})
            events.append({
                "id": e.get("id"),
                "summary": e.get("summary", "(No title)"),
                "start": start.get("dateTime", start.get("date")),
                "end": end.get("dateTime", end.get("date")),
                "status": e.get("status"),
            })
        return {"date": date_str, "timezone": tz_name, "events": events}
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def check_availability(
    token_data: dict,
    start_datetime: str,
    end_datetime: str,
    timezone_str: Optional[str] = None,
) -> dict:
    """Tool: Check whether a time window is free on primary calendar."""
    tz, tz_name = _get_tz(timezone_str or DEFAULT_TIMEZONE)
    try:
        def _parse(s):
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt

        start_dt = _parse(start_datetime)
        end_dt = _parse(end_datetime)

        service = _get_service(token_data)
        body = {
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "timeZone": tz_name,
            "items": [{"id": "primary"}],
        }
        result = service.freebusy().query(body=body).execute()
        busy = result.get("calendars", {}).get("primary", {}).get("busy", [])
        if busy:
            conflicts = [{"start": b["start"], "end": b["end"]} for b in busy]
            return {
                "available": False,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "conflicts": conflicts,
            }
        return {
            "available": True,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "conflicts": [],
        }
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def create_calendar_event(
    token_data: dict,
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: Optional[str] = None,
    attendees: Optional[list] = None,
    timezone_str: Optional[str] = None,
) -> dict:
    """Tool: Create a confirmed Google Calendar event."""
    tz, tz_name = _get_tz(timezone_str or DEFAULT_TIMEZONE)
    try:
        def _parse(s):
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            return dt

        start_dt = _parse(start_datetime)
        end_dt = _parse(end_datetime)

        event_body = {
            "summary": summary,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": tz_name},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": tz_name},
        }
        if description:
            event_body["description"] = description
        if attendees:
            event_body["attendees"] = [{"email": a} for a in attendees if "@" in str(a)]

        service = _get_service(token_data)
        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return {
            "success": True,
            "event_id": created.get("id"),
            "summary": created.get("summary"),
            "start": created.get("start", {}).get("dateTime"),
            "end": created.get("end", {}).get("dateTime"),
            "html_link": created.get("htmlLink"),
            "status": created.get("status"),
        }
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def get_event_details(token_data: dict, event_id: str) -> dict:
    """Tool: Retrieve a specific event by ID to verify creation."""
    try:
        service = _get_service(token_data)
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        return {
            "event_id": event.get("id"),
            "summary": event.get("summary"),
            "start": event.get("start", {}).get("dateTime", event.get("start", {}).get("date")),
            "end": event.get("end", {}).get("dateTime", event.get("end", {}).get("date")),
            "status": event.get("status"),
            "html_link": event.get("htmlLink"),
            "attendees": [a.get("email") for a in event.get("attendees", [])],
            "description": event.get("description", ""),
        }
    except HttpError as e:
        return {"error": f"Google Calendar API error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}
