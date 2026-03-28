"""
bot/__init__.py — Core data contracts for the Quorum voice-bot layer.

These two dataclasses are the shared language between every module in this
package. Import from here — never redefine them elsewhere.

TranscriptSegment is the exact shape the agent branch expects. Any change
here must be coordinated with agent/orchestrator.py.
"""

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """
    A single unit of transcribed speech from Deepgram.

    Sent to the agent orchestrator on every transcript event. The agent
    ignores segments where is_final is False — they are used only for
    real-time display purposes (e.g. a companion web app).

    Fields:
        text:       Clean, punctuated spoken words from Deepgram smart_format.
        speaker:    Diarization label e.g. "Speaker 0", "Speaker 1".
                    Stays consistent for the lifetime of the meeting session.
        timestamp:  Unix timestamp of when this segment was captured.
        is_final:   False = interim word-by-word result (speculative).
                    True  = committed, punctuated final transcript.
        meeting_id: Unique ID for this meeting session, consistent across
                    all segments from the same call.
    """
    text: str
    speaker: str
    timestamp: float
    is_final: bool
    meeting_id: str


@dataclass
class BotStatus:
    """
    Lifecycle state of the Recall.ai meeting bot.

    Returned by RecallClient methods and emitted on webhook events so the
    rest of the system knows what the bot is doing.

    Fields:
        meeting_id:  Unique ID for this meeting session.
        status:      One of: "joining" | "active" | "left" | "error"
        bot_id:      The Recall.ai bot UUID — used for all subsequent API calls.
        meeting_url: The original meeting URL the bot was sent to.
        joined_at:   Unix timestamp of when the bot entered the call, or None
                     if it hasn't joined yet.
    """
    meeting_id: str
    status: str
    bot_id: str
    meeting_url: str
    joined_at: float | None


__all__ = ["TranscriptSegment", "BotStatus"]
