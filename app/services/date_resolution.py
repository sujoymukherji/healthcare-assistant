from __future__ import annotations

import re
from datetime import date, datetime, timedelta


_WEEKDAY_MAP = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6,
}


def resolve_single_date(query: str, *, prefer_first: bool = False) -> str | None:
    iso_matches = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', query)
    if iso_matches:
        return iso_matches[0] if prefer_first else iso_matches[-1]

    lowered = query.lower()
    today = date.today()
    if 'today' in lowered:
        return today.isoformat()
    if 'tomorrow' in lowered:
        return (today + timedelta(days=1)).isoformat()

    next_week_match = re.search(
        r'\bnext week\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        lowered,
    )
    if next_week_match:
        weekday = _WEEKDAY_MAP[next_week_match.group(1)]
        next_week_monday = today + timedelta(days=(7 - today.weekday()))
        return (next_week_monday + timedelta(days=weekday)).isoformat()

    next_weekday_match = re.search(
        r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        lowered,
    )
    if next_weekday_match:
        weekday = _WEEKDAY_MAP[next_weekday_match.group(1)]
        days_ahead = (weekday - today.weekday()) % 7
        days_ahead = 7 if days_ahead == 0 else days_ahead
        return (today + timedelta(days=days_ahead)).isoformat()

    weekday_match = re.search(
        r'\b(?:on|for|from|this)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        lowered,
    )
    if weekday_match:
        weekday = _WEEKDAY_MAP[weekday_match.group(1)]
        days_ahead = (weekday - today.weekday()) % 7
        return (today + timedelta(days=days_ahead)).isoformat()

    natural_matches = re.findall(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)(?:\s+(20\d{2}))?\b', query)
    if not natural_matches:
        return None
    chosen = natural_matches[0] if prefer_first else natural_matches[-1]
    day_text, month_text, year_text = chosen
    year = int(year_text) if year_text else datetime.now().year
    for fmt in ('%d %B %Y', '%d %b %Y'):
        try:
            return datetime.strptime(f'{int(day_text):02d} {month_text} {year}', fmt).date().isoformat()
        except ValueError:
            continue
    return None


def resolve_date_or_range(query: str) -> tuple[str | None, str | None]:
    lowered = query.lower()
    today = date.today()
    if 'next week' in lowered and not re.search(
        r'\bnext week\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        lowered,
    ):
        next_monday = today + timedelta(days=(7 - today.weekday()))
        next_saturday = next_monday + timedelta(days=5)
        return next_monday.isoformat(), next_saturday.isoformat()
    if 'this week' in lowered and not re.search(
        r'\bthis week\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        lowered,
    ):
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=5)
        return week_start.isoformat(), week_end.isoformat()
    if 'next month' in lowered:
        next_month_year = today.year + (1 if today.month == 12 else 0)
        next_month = 1 if today.month == 12 else today.month + 1
        start = date(next_month_year, next_month, 1)
        following_year = next_month_year + (1 if next_month == 12 else 0)
        following_month = 1 if next_month == 12 else next_month + 1
        end = date(following_year, following_month, 1) - timedelta(days=1)
        return start.isoformat(), end.isoformat()
    return resolve_single_date(query), None
