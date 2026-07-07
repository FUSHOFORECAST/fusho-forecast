from src.forecast_candidates.memory_candidates import (
    MemoryShortCandidate,
    MemoryMediumCandidate,
    MemoryLongCandidate,
    AdaptiveMemoryCandidate,
)

from src.forecast_candidates.context_candidates import (
    StateForecastCandidate,
    CalendarForecastCandidate,
    SimilarityForecastCandidate,
)

from src.forecast_candidates.external_context_candidate import (
    ExternalContextCandidate,
)


CANDIDATE_REGISTRY = [
    MemoryShortCandidate(),
    MemoryMediumCandidate(),
    MemoryLongCandidate(),
    AdaptiveMemoryCandidate(),
    StateForecastCandidate(),
    CalendarForecastCandidate(),
    SimilarityForecastCandidate(),
    ExternalContextCandidate(),
]


def get_candidate_names():
    return [candidate.name for candidate in CANDIDATE_REGISTRY]
