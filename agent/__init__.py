"""
agent — Quorum intelligence layer

Public API for the agent package. Import from here rather than from
individual modules so other branches and tests have a stable surface.

Cross-branch interfaces (use these dataclasses when sending data to/from
the agent):
    TranscriptSegment  — voice-bot → orchestrator
    SpeakCommand       — orchestrator → voice-bot
    ContextRequest     — orchestrator → integrations
    IntegrationResult  — integrations → orchestrator
    ActionRequest      — orchestrator → actions-ui
"""

from .context import MeetingContext
from .intent import Intent, IntentDetector
from .mode import ModeManager
from .orchestrator import (
    ActionRequest,
    ContextRequest,
    IntegrationResult,
    QuorumOrchestrator,
    SpeakCommand,
    TranscriptSegment,
)

__all__ = [
    # Main orchestrator — entry point for voice-bot integration
    "QuorumOrchestrator",
    # Cross-branch dataclasses
    "TranscriptSegment",
    "SpeakCommand",
    "ContextRequest",
    "IntegrationResult",
    "ActionRequest",
    # Sub-components (useful for testing individual layers)
    "IntentDetector",
    "Intent",
    "MeetingContext",
    "ModeManager",
]
