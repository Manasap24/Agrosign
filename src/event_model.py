"""Sentence-level agricultural event representation."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgriculturalEvent:
    process_id: str
    process_name: str
    action: str = ""
    agent: str = ""
    patient: str = ""
    method: str = ""
    polarity: str = "positive"
    polarity_evidence: str = ""
    status: str = "asserted"
    tense: str = "unknown"
    sequence_index: int = 1
    temporal_links: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.0
    confidence_level: str = "low"
    context_verified: bool = False
    process_variant_id: str = ""
    category: str = ""
    sign_concepts: list[str] = field(default_factory=list)
    semantic_confidence: float = 0.0
    bart_verified: bool | None = None
    bart_score: float | None = None
    sequence_relation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
