import json
import os
from typing import Any

from groq import Groq
from dotenv import load_dotenv

from .models import Intent, IntentType

load_dotenv()

_SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.

Analyze the user's transcribed speech and return a JSON object containing an array of intents.

## Supported Intent Types
- create_file: user wants to create/save a file.
  params: {"filename": "<name>", "content": "<optional text content>"}
- write_code: user wants to generate code and write it to a file.
  params: {"language": "<python|javascript|etc>", "description": "<what the code should do>", "filename": "<output filename>"}
- summarize: user wants to summarize a block of text.
  params: {"text": "<text to summarize>"}
- general_chat: user wants to have a conversation or ask a question.
  params: {"message": "<user's message>"}
- unknown: intent is unclear, unsupported, or confidence is low.
  params: {}

## Compound Commands
Multiple intents are supported in a single utterance. For example:
"Summarize this text and save it to summary.txt" →
  [{"intent_type": "summarize", "params": {"text": "..."}, "confidence": 0.95},
   {"intent_type": "create_file", "params": {"filename": "summary.txt"}, "confidence": 0.95}]

Note: When create_file follows summarize, omit "content" — the system will pipe the summary automatically.

## Output Format
Return ONLY this JSON structure, nothing else:
{
  "intents": [
    {"intent_type": "...", "params": {...}, "confidence": 0.0}
  ]
}

Use confidence < 0.5 for uncertain intents and set intent_type to "unknown"."""


def classify_intent(transcription: str, history: list[dict[str, Any]] | None = None) -> list[Intent]:
    """Classify a transcription into one or more intents using Groq.

    Args:
        transcription: The STT output text.
        history: Optional list of prior {user, assistant} dicts for context.

    Returns:
        List of Intent objects. Falls back to [unknown] on parse failure.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject up to 5 prior turns for compound/contextual commands
    if history:
        for entry in history[-5:]:
            if entry.get("user"):
                messages.append({"role": "user", "content": entry["user"]})
            if entry.get("assistant"):
                messages.append({"role": "assistant", "content": entry["assistant"]})

    messages.append({"role": "user", "content": transcription})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={"type": "json_object"},  # JSON Object Mode (not strict schema)
        temperature=0.1,
        max_tokens=512,
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        intents_data = data.get("intents", [])
        intents = []
        for item in intents_data:
            try:
                intents.append(Intent.model_validate(item))
            except Exception:
                continue
        if intents:
            return intents
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: return a single unknown intent
    return [Intent(intent_type="unknown", params={}, confidence=0.0)]
