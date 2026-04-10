# Research Package: Voice-Controlled Local AI Agent

**Stack:** Python 3.12 · Apple Silicon M-series · uv · faster-whisper · groq · streamlit ≥1.32 · pydantic v2 · python-dotenv

---

## 1. faster-whisper

**Package:** `faster-whisper` (latest: 1.2.1, Oct 2025)  
**Backend:** CTranslate2 — **does NOT support Apple Silicon MPS/Metal**

### Import

```python
from faster_whisper import WhisperModel
```

### Constructor

```python
WhisperModel(
    model_size_or_path: str,   # "tiny","base","small","medium","large-v3","turbo", or local path/HF repo
    device: str = "auto",       # "cpu" | "cuda" | "auto"  — NO "mps" support
    device_index: int | list[int] = 0,
    compute_type: str = "default",  # see table below
    cpu_threads: int = 0,      # 0 = use all cores
    num_workers: int = 1,
    download_root: str | None = None,
    local_files_only: bool = False,
)
```

**`compute_type` on Apple Silicon CPU:**

| compute_type | Speed | RAM | Quality | Use on Apple Silicon |
|---|---|---|---|---|
| `"int8"` | fastest | lowest | good | ✅ recommended |
| `"int8_float32"` | fast | medium | better | ✅ ok |
| `"float32"` | slow | highest | best | ✅ ok (debug) |
| `"float16"` | — | — | — | ❌ CPU int8 preferred |

**Apple Silicon correct instantiation:**
```python
# NO MPS — CTranslate2 has no Metal/MPS backend
# Use device="cpu" with int8
model = WhisperModel("base", device="cpu", compute_type="int8")
```

### transcribe() signature

```python
segments, info = model.transcribe(
    audio,                          # str | Path | bytes | BinaryIO
    language: str | None = None,    # "en", "fr", etc. None = auto-detect
    task: str = "transcribe",       # "transcribe" | "translate"
    beam_size: int = 5,
    best_of: int = 5,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    temperature: float | list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold: float = 2.4,
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = True,
    word_timestamps: bool = False,
    vad_filter: bool = False,
    vad_parameters: dict | None = None,
    # ... more
)
# Returns: Tuple[Generator[Segment], TranscriptionInfo]
```

**Return types:**
```python
# info is available immediately (no iteration needed)
info.language          # str, e.g. "en"
info.language_probability  # float 0.0-1.0
info.duration          # float, seconds

# segments is a LAZY GENERATOR — transcription runs on iteration
for segment in segments:
    segment.start   # float, seconds
    segment.end     # float, seconds
    segment.text    # str
    segment.words   # list[Word] if word_timestamps=True
```

### ⚠️ Critical Gotcha: Lazy Generator

```python
# WRONG — info.language is fine but segments haven't run yet
segments, info = model.transcribe("audio.wav")
text = " ".join([s.text for s in segments])  # transcription runs HERE

# CORRECT pattern for getting full text
segments, info = model.transcribe("audio.wav")
segments_list = list(segments)  # force execution NOW
text = " ".join(s.text for s in segments_list)
```

### Transcribe from BytesIO (mic input)

```python
import io

def transcribe_bytes(model: WhisperModel, audio_bytes: bytes) -> str:
    audio_io = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(audio_io, language="en")
    return " ".join(s.text.strip() for s in segments)
```

### Model sizes for demo

| Size | Params | Disk | Speed (CPU int8) | Recommended |
|---|---|---|---|---|
| `"tiny"` | 39M | ~75MB | very fast | accuracy too low |
| `"base"` | 74M | ~145MB | fast | ✅ best for demo |
| `"small"` | 244M | ~465MB | moderate | ✅ better accuracy |
| `"medium"` | 769M | ~1.5GB | slow | overkill for demo |

**Recommendation:** `"base"` for demo speed; `"small"` for better accuracy.

### First-run download

Models auto-download from HuggingFace to `~/.cache/huggingface/hub/`. Base model: ~30s on good connection. Subsequent runs load from cache.

---

## 2. groq Python SDK

**Package:** `groq` (latest: **1.1.2**, released ~2 weeks ago)  
**Requires:** Python ≥ 3.10

### Import & Client

```python
from groq import Groq
import os

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  # auto-reads GROQ_API_KEY env var
    timeout=30.0,      # optional, default 60s
    max_retries=2,     # optional, default 2
)
```

### chat.completions.create() signature

```python
response = client.chat.completions.create(
    model: str,                          # required
    messages: list[dict],                # required
    temperature: float = 1.0,            # 0.0–2.0
    max_tokens: int | None = None,       # None = model max
    top_p: float = 1.0,
    stream: bool = False,
    stop: str | list[str] | None = None,
    response_format: dict | None = None, # for JSON mode
    seed: int | None = None,
    # ...
)

# Response access
response.choices[0].message.content  # str
response.usage.prompt_tokens          # int
response.usage.completion_tokens      # int
response.usage.total_tokens           # int
```

### Message format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},  # for multi-turn
]
```

### Available fast models

| Model ID | Context | Speed | Notes |
|---|---|---|---|
| `"llama-3.3-70b-versatile"` | 128K | fast | ✅ best balance for this project |
| `"llama-3.1-8b-instant"` | 128K | fastest | lower quality |
| `"meta-llama/llama-4-scout-17b-16e-instruct"` | 128K | fast | newer |

### JSON Object Mode (works with llama-3.3-70b-versatile)

```python
import json

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": """Classify the intent. Return ONLY valid JSON:
{"intent": "create_file|write_code|summarize|general_chat|unknown", "params": {...}}"""
        },
        {"role": "user", "content": user_text}
    ],
    response_format={"type": "json_object"},
    temperature=0.1,  # low temp for deterministic JSON
)

data = json.loads(response.choices[0].message.content)
```

### Structured Output (json_schema — strict mode)

Only available on `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. For llama models, use JSON Object Mode above.

```python
# Pydantic schema → JSON Schema (for supported models only)
from pydantic import BaseModel
import json

class Intent(BaseModel):
    intent: str
    confidence: float

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "intent_classification",
            "strict": True,
            "schema": Intent.model_json_schema()
        }
    }
)
result = Intent.model_validate(json.loads(response.choices[0].message.content))
```

### Rate Limits (Free Tier — as of April 2026)

| Model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
| `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | 500,000 |

Rate limit hit → `groq.RateLimitError` (HTTP 429). SDK auto-retries 2x with backoff.

### Error handling

```python
import groq

try:
    response = client.chat.completions.create(...)
except groq.RateLimitError:
    # 429 — back off
except groq.APIConnectionError as e:
    # network issue
except groq.APIStatusError as e:
    print(e.status_code, e.response)
```

---

## 3. Streamlit

### st.audio_input()

**Introduced:** Streamlit 1.31 (experimental) → **stable in 1.32**

```python
audio_data = st.audio_input(
    label: str,                          # required display label
    *,
    key: str | int | None = None,        # widget identity key
    help: str | None = None,
    on_change: Callable | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    disabled: bool = False,
    label_visibility: str = "visible",   # "visible"|"hidden"|"collapsed"
)
# Returns: UploadedFile | None
```

**Return value:** An `UploadedFile` (file-like object) or `None` if no recording yet.

```python
audio_data = st.audio_input("Record your voice")
if audio_data is not None:
    audio_bytes = audio_data.read()      # bytes (WAV format)
    # or
    audio_bytes = audio_data.getvalue()  # bytes (non-destructive)
    audio_data.name   # str, e.g. "audio.wav"
    audio_data.type   # str, e.g. "audio/wav"
```

### st.file_uploader() for audio

```python
uploaded = st.file_uploader(
    label="Upload audio",
    type=["wav", "mp3", "ogg", "m4a"],
    accept_multiple_files=False,
    key="audio_upload",
)
if uploaded is not None:
    audio_bytes = uploaded.getvalue()
```

### st.audio() — playback

```python
# Play uploaded or recorded audio back
st.audio(audio_bytes, format="audio/wav")
# or
st.audio(audio_data)  # pass UploadedFile directly
```

### st.session_state

```python
# Initialize
if "history" not in st.session_state:
    st.session_state.history = []

# Read
items = st.session_state.history

# Write
st.session_state.history.append(new_item)

# Reset
st.session_state.history = []
```

### st.tabs()

```python
tab1, tab2 = st.tabs(["Microphone", "Upload File"])

with tab1:
    audio_data = st.audio_input("Record")

with tab2:
    uploaded = st.file_uploader("Upload", type=["wav", "mp3"])
```

### st.sidebar

```python
with st.sidebar:
    st.header("Session History")
    if st.button("Clear History"):
        st.session_state.history = []
    for entry in st.session_state.get("history", []):
        st.text(entry["transcription"][:80])
```

### st.expander()

```python
with st.expander("View full transcription", expanded=False):
    st.write(transcription_text)
```

### Status / feedback widgets

```python
with st.spinner("Transcribing audio..."):
    result = transcribe(audio_bytes)

st.success("Done! Transcribed successfully.")
st.error("Error: Could not transcribe audio.")
st.info("Tip: Speak clearly for best results.")
st.warning("Audio too short — minimum 1 second required.")
```

### st.code()

```python
st.code(generated_code, language="python")
st.code(json_output, language="json")
```

### ⚠️ Streamlit Gotchas

1. **Re-runs on every interaction.** All code runs top-to-bottom on every widget change. Use `st.session_state` to persist data.
2. **`st.audio_input` returns `None` on first load** before user records.
3. **`uploaded.read()` is destructive** — call `.read()` once, store bytes; or use `.getvalue()` which is repeatable.
4. **Tabs re-run on switch** — store results in `session_state`, not local variables.

---

## 4. Pydantic v2

### BaseModel definition

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
```

### Model with Literal enum-like fields

```python
class Intent(BaseModel):
    intent_type: Literal[
        "create_file", "write_code", "summarize",
        "general_chat", "unknown"
    ]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    params: dict = Field(default_factory=dict)
```

### Optional fields

```python
class ToolResult(BaseModel):
    success: bool
    output: str
    file_path: Optional[str] = None    # old style (still valid)
    error: str | None = None           # new style (preferred in Python 3.10+)
```

### Field() with defaults

```python
class HistoryEntry(BaseModel):
    timestamp: str
    transcription: str = Field(default="", description="Raw STT output")
    intent_type: str = Field(default="unknown")
    result: str = Field(default="")
    audio_source: Literal["microphone", "upload"] = "microphone"
```

### model_validate() and model_dump()

```python
# v2 equivalents of v1's parse_obj() and dict()
data = {"intent_type": "create_file", "params": {"filename": "hello.txt"}}

# Validate from dict
intent = Intent.model_validate(data)

# Validate from JSON string
intent = Intent.model_validate_json('{"intent_type": "create_file", "params": {}}')

# Dump to dict
d = intent.model_dump()
# {'intent_type': 'create_file', 'confidence': 1.0, 'params': {}}

# Dump to JSON string
j = intent.model_dump_json()
```

### JSON schema for structured outputs

```python
# Generate schema for Groq structured output
schema = Intent.model_json_schema()
```

### ⚠️ Pydantic v2 Breaking Changes from v1

| v1 | v2 |
|---|---|
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `parse_obj()` | `model_validate()` |
| `parse_raw()` | `model_validate_json()` |
| `schema()` | `model_json_schema()` |
| `validator` decorator | `field_validator` |

---

## 5. pyproject.toml with uv

### Structure

```toml
[project]
name = "memo-al"
version = "0.1.0"
description = "Voice-controlled local AI agent"
requires-python = ">=3.12"
dependencies = [
    "faster-whisper>=1.0.0",
    "groq>=1.1.0",
    "streamlit>=1.32.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### uv commands

```bash
# Install all dependencies
uv sync

# Add a new dependency (updates pyproject.toml + lockfile)
uv add faster-whisper
uv add "groq>=1.1.0"
uv add "streamlit>=1.32"

# Run in uv environment
uv run streamlit run app.py
uv run python main.py

# Lock without installing
uv lock
```

### Exact PyPI package names

| Library | PyPI name | Import |
|---|---|---|
| Faster Whisper | `faster-whisper` | `from faster_whisper import WhisperModel` |
| Groq SDK | `groq` | `from groq import Groq` |
| Streamlit | `streamlit` | `import streamlit as st` |
| python-dotenv | `python-dotenv` | `from dotenv import load_dotenv` |
| Pydantic v2 | `pydantic` | `from pydantic import BaseModel` |

---

## 6. Path Safety (prevent path traversal)

### Core pattern

```python
from pathlib import Path

BASE_DIR = Path("output").resolve()  # absolute, e.g. /project/output

def safe_path(filename: str) -> Path:
    """
    Resolve user-supplied filename under BASE_DIR.
    Raises ValueError if path escapes the sandbox.
    """
    # Strip leading slashes/dots that could escape
    # Path(filename).name strips all directory components
    safe_name = Path(filename).name          # "../../etc/passwd" → "passwd"
    resolved = (BASE_DIR / safe_name).resolve()

    # Double-check it's still under BASE_DIR
    if not resolved.is_relative_to(BASE_DIR):
        raise ValueError(f"Path traversal detected: {filename!r}")

    return resolved
```

### If subdirectories are allowed

```python
def safe_subpath(user_path: str) -> Path:
    """Allow subdirectories but still sandbox under BASE_DIR."""
    candidate = (BASE_DIR / user_path).resolve()
    if not candidate.is_relative_to(BASE_DIR):
        raise ValueError(f"Path escapes sandbox: {user_path!r}")
    # Ensure parent directories exist
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate
```

### ⚠️ Gotchas

- `Path.is_relative_to()` requires Python ≥ 3.9. On older: use `str(resolved).startswith(str(BASE_DIR))`.
- Always call `.resolve()` on **both** the base and the candidate — symlinks can escape without it.
- `Path(filename).name` removes ALL directory components — use only when subdirs are not needed.
- `Path.resolve()` calls the OS — file doesn't need to exist (but it resolves to absolute).

---

## 7. Integration Patterns for This Project

### STT module (agent/stt.py)

```python
import io
from functools import lru_cache
from faster_whisper import WhisperModel

@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    """Singleton — loads once, reused across requests."""
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe(audio_bytes: bytes) -> str:
    model = get_model()
    audio_io = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(audio_io, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments)
    if not text.strip():
        raise ValueError("No speech detected in audio.")
    return text
```

### Intent classifier (agent/intent.py)

```python
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = Groq()  # auto-reads GROQ_API_KEY

SYSTEM_PROMPT = """You are an intent classifier. Given a voice command, return ONLY valid JSON:
{
  "intents": [
    {
      "intent_type": "create_file|write_code|summarize|general_chat|unknown",
      "confidence": 0.0-1.0,
      "params": {}
    }
  ]
}
Params for create_file: {"filename": str, "content": str}
Params for write_code: {"language": str, "description": str, "filename": str}
Params for summarize: {"text": str}
Params for general_chat: {"message": str}
Support compound commands (multiple intents in the array)."""

def classify(transcription: str) -> dict:
    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)
```

### Streamlit UI snippet (app.py)

```python
import streamlit as st
from agent.stt import transcribe
from agent.intent import classify

st.title("MemO-Al: Voice Agent")

if "history" not in st.session_state:
    st.session_state.history = []

tab_mic, tab_upload = st.tabs(["Microphone", "Upload File"])

audio_bytes = None

with tab_mic:
    audio_data = st.audio_input("Click to record")
    if audio_data:
        audio_bytes = audio_data.getvalue()
        st.audio(audio_data)

with tab_upload:
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3"])
    if uploaded:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

if audio_bytes and st.button("Process", type="primary"):
    with st.spinner("Transcribing..."):
        try:
            text = transcribe(audio_bytes)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.info(f"**Transcription:** {text}")

    with st.spinner("Classifying intent..."):
        result = classify(text)

    st.json(result)
```

---

## 8. Known Gotchas Summary

| Library | Gotcha | Fix |
|---|---|---|
| faster-whisper | No MPS/Metal support on Apple Silicon | Use `device="cpu"` |
| faster-whisper | `segments` is a lazy generator | Call `list(segments)` or iterate once |
| faster-whisper | Model downloads on first run (~30-60s) | Show spinner; cache with `@lru_cache` |
| groq | `llama-3.3-70b-versatile` doesn't support `strict` json_schema | Use `{"type": "json_object"}` + system prompt |
| groq | Free tier: 30 RPM, 1K RPD for 70b model | Catch `RateLimitError` |
| streamlit | All code re-runs on every interaction | Use `st.session_state` for persistence |
| streamlit | `uploaded.read()` is destructive (cursor advances) | Use `.getvalue()` instead |
| streamlit | `st.audio_input` returns `None` until user records | Guard with `if audio_data is not None` |
| pydantic v2 | `.dict()` and `.parse_obj()` removed | Use `.model_dump()` and `.model_validate()` |
| pathlib | `.resolve()` needed to catch symlink escapes | Always resolve both base and candidate |
