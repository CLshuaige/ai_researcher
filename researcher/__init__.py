from researcher.researcher import AIResearcher
#from researcher.config import config
from researcher.schemas import (
    ResearchIdea,
    ExperimentalMethod,
    ExperimentResult,
    ReviewReport,
    LiteratureReview
)

__version__ = "0.1.0"
__all__ = [
    "AIResearcher",
    #"config",
    "ResearchIdea",
    "ExperimentalMethod",
    "ExperimentResult",
    "ReviewReport",
    "LiteratureReview"
]
