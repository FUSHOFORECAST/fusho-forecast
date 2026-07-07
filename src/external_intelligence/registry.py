from src.external_intelligence.calendar_engine import run_calendar_engine
from src.external_intelligence.events_engine import run_events_engine


EXTERNAL_ENGINES = [
    run_calendar_engine,
    run_events_engine,
]
