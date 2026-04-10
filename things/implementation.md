# MemO-Al — Voice-Controlled Local AI Agent

## Goal
Build a full-stack voice-controlled AI agent: Faster-Whisper (local STT) → Groq intent classification → tool execution (file/code/summarize/chat) → Streamlit UI, running on Apple Silicon with session memory and compound command support.

## Prerequisites
Ensure you are working in the project root:
```
/Users/shubhmac/Developer/Hiring Assignments/MemO-Al
```
All commands are run from this directory with `uv`.

---

### Step-by-Step Instructions

---

## Step 1: Project Setup

- [ ] Replace the contents of `pyproject.toml` with the code below:

```toml
[project]
name = "memo-al"
version = "0.1.0"
description = "Voice-Controlled Local AI Agent — STT → Intent → Tool Execution"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "docling>=2.85.0",
    "docling-core>=2.73.0",
    "langchain-core>=1.2.28",
    "faster-whisper>=1.0.3",
    "groq>=0.9.0",
    "streamlit>=1.32.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]
```

- [ ] Create `.env.example` with the contents below:

```
GROQ_API_KEY=your_groq_api_key_here
```

- [ ] Create `.env` (copy from `.env.example`) and add your real Groq API key:

```bash
cp .env.example .env
# Then open .env and replace the placeholder with your key from https://console.groq.com
```

- [ ] Create `.gitignore` with the contents below:

```
# Environment
.env

# Output folder contents (keep the folder, ignore contents)
output/*
!output/.gitkeep

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
dist/
*.egg-info/

# Streamlit
.streamlit/

# macOS
.DS_Store
```

- [ ] Create the `agent/` package by creating `agent/__init__.py` with the contents below:

```python
```

- [ ] Create the `output/` folder placeholder:

```bash
mkdir -p output && touch output/.gitkeep
```

- [ ] Run dependency install:

```bash
uv sync
```

### Step 1 Verification Checklist
- [ ] `uv sync` completes with no errors
- [ ] `agent/` directory exists with `__init__.py` inside
- [ ] `output/` directory exists with `.gitkeep` inside
- [ ] `.env` exists and contains your `GROQ_API_KEY`
- [ ] `.env.example` exists and does not contain the real key

### Step 1 STOP & COMMIT
**STOP & COMMIT:** Stop here. Test that `uv sync` succeeds, then stage and commit with message: `feat: project setup — deps, env, agent package, output dir`

---

## Step 2: STT Module

- [ ] Create `agent/stt.py` with the contents below:

```python
import io
from typing import Union

from faster_whisper import WhisperModel

# Module-level singleton — model loads once per process (~30–60s on first run)
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load the Faster-Whisper base model.

    Apple Silicon note: CTranslate2 (used by faster-whisper) does not support
    Apple's Metal (MPS) backend. We use device="cpu" with int8 quantization,
    which is the correct and fastest option for M-series chips.
    """
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_input: Union[str, io.BytesIO]) -> str:
    """Transcribe audio to text using Faster-Whisper (local, offline).

    Args:
        audio_input: File path string OR BytesIO object containing raw audio bytes.

    Returns:
        Transcribed text as a single string.

    Raises:
        RuntimeError: If no speech is detected or transcription fails.
    """
    model = _get_model()

    # vad_filter=True silences non-speech segments before transcription
    segments, _info = model.transcribe(
        audio_input,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # segments is a LAZY generator — iteration triggers the actual transcription
    text = " ".join(segment.text.strip() for segment in segments).strip()

    if not text:
        raise RuntimeError(
            "No speech detected. Ensure the audio contains clear speech and is not silent."
        )

    return text
```

### Step 2 Verification Checklist
- [ ] No import errors: `uv run python -c "from agent.stt import transcribe_audio; print('OK')`
- [ ] (Optional) Quick smoke test with a `.wav` file you have:
  ```bash
  uv run python -c "
  from agent.stt import transcribe_audio
  result = transcribe_audio('path/to/test.wav')
  print(result)
  "
  ```
  Expected: printed transcription string. First run downloads ~145MB model.

### Step 2 STOP & COMMIT
**STOP & COMMIT:** Stop here. Verify the import works, then stage and commit: `feat: STT module — Faster-Whisper base model, CPU/int8, VAD filter`

---

## Step 3: Intent Classification

- [ ] Create `agent/models.py` with the contents below:

```python
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# All supported intent types
IntentType = Literal["create_file", "write_code", "summarize", "general_chat", "unknown"]


class Intent(BaseModel):
    """A single classified intent with extracted parameters."""

    intent_type: IntentType
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ToolResult(BaseModel):
    """The result of executing one intent/tool."""

    intent_type: IntentType
    success: bool
    output: str
    file_path: Optional[str] = None


class PipelineResult(BaseModel):
    """Full result of one voice command pipeline run."""

    transcription: str
    intents: list[Intent]
    results: list[ToolResult]
```

- [ ] Create `agent/intent.py` with the contents below:

```python
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
```

### Step 3 Verification Checklist
- [ ] No import errors: `uv run python -c "from agent.intent import classify_intent; print('OK')"`
- [ ] Quick smoke test (requires `GROQ_API_KEY` in `.env`):
  ```bash
  uv run python -c "
  from agent.intent import classify_intent
  result = classify_intent('Create a file called hello.txt')
  print(result)
  "
  ```
  Expected output: `[Intent(intent_type='create_file', params={'filename': 'hello.txt', ...}, confidence=...)]`

### Step 3 STOP & COMMIT
**STOP & COMMIT:** Stop here. Verify both imports work and the smoke test returns a parsed intent, then commit: `feat: intent classification — Groq JSON mode, compound command support, pydantic models`

---

## Step 4: Tool Execution

- [ ] Create `agent/tools.py` with the contents below:

```python
import os
from pathlib import Path
from typing import Any

from groq import Groq
from dotenv import load_dotenv

from .models import Intent, IntentType, ToolResult

load_dotenv()

# All file operations are restricted to this directory — no exceptions
_OUTPUT_DIR = Path("output").resolve()


def _get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _safe_path(filename: str) -> Path:
    """Resolve a filename safely under OUTPUT_DIR.

    Uses only the final path component (Path.name) to strip any directory
    prefix, then verifies the resolved path is still under OUTPUT_DIR.

    Raises:
        ValueError: If path traversal is detected.
    """
    safe_name = Path(filename).name  # strips any "../../" prefix
    resolved = (_OUTPUT_DIR / safe_name).resolve()
    if not resolved.is_relative_to(_OUTPUT_DIR):
        raise ValueError(f"Path traversal detected in filename: {filename!r}")
    return resolved


# ---------------------------------------------------------------------------
# Individual tool functions
# ---------------------------------------------------------------------------

def create_file(filename: str, content: str = "") -> ToolResult:
    """Write content to a file inside output/."""
    try:
        path = _safe_path(filename)
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            intent_type="create_file",
            success=True,
            output=f"File created: output/{path.name}",
            file_path=str(path),
        )
    except Exception as e:
        return ToolResult(intent_type="create_file", success=False, output=str(e))


def write_code(language: str, description: str, filename: str) -> ToolResult:
    """Generate code via Groq and write it to output/filename."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert programmer. Write clean, working, well-structured code. "
                        "Return ONLY the raw code — no explanations, no markdown code fences, no comments "
                        "unless the code itself requires them."
                    ),
                },
                {"role": "user", "content": f"Write {language} code for: {description}"},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        code = response.choices[0].message.content.strip()

        path = _safe_path(filename)
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

        return ToolResult(
            intent_type="write_code",
            success=True,
            output=code,
            file_path=str(path),
        )
    except Exception as e:
        return ToolResult(intent_type="write_code", success=False, output=str(e))


def summarize(text: str) -> ToolResult:
    """Summarize text via Groq."""
    try:
        if not text.strip():
            return ToolResult(
                intent_type="summarize",
                success=False,
                output="No text provided to summarize.",
            )
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise, accurate summarizer. Summarize clearly in 2–4 sentences.",
                },
                {"role": "user", "content": f"Summarize the following:\n\n{text}"},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        summary = response.choices[0].message.content.strip()
        return ToolResult(intent_type="summarize", success=True, output=summary)
    except Exception as e:
        return ToolResult(intent_type="summarize", success=False, output=str(e))


def general_chat(message: str, history: list[dict[str, Any]] | None = None) -> ToolResult:
    """Answer a conversational message via Groq with optional history."""
    try:
        client = _get_client()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful, concise AI assistant."}
        ]

        if history:
            for entry in history[-10:]:
                if entry.get("user"):
                    messages.append({"role": "user", "content": entry["user"]})
                if entry.get("assistant"):
                    messages.append({"role": "assistant", "content": entry["assistant"]})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = response.choices[0].message.content.strip()
        return ToolResult(intent_type="general_chat", success=True, output=reply)
    except Exception as e:
        return ToolResult(intent_type="general_chat", success=False, output=str(e))


# ---------------------------------------------------------------------------
# Compound executor
# ---------------------------------------------------------------------------

def execute_intents(
    intents: list[Intent],
    history: list[dict[str, Any]] | None = None,
) -> list[ToolResult]:
    """Execute a list of intents in sequence.

    Compound command support: the output of one successful step is passed as
    implicit content to the next step if it requires it (e.g. summarize → create_file).
    """
    results: list[ToolResult] = []
    prior_output: str | None = None

    for intent in intents:
        p = intent.params
        result: ToolResult

        if intent.intent_type == "create_file":
            filename = p.get("filename", "output.txt")
            # Use explicit content if provided, otherwise pipe prior step output
            content = p.get("content") or prior_output or ""
            result = create_file(filename, content)

        elif intent.intent_type == "write_code":
            language = p.get("language", "python")
            description = p.get("description", "a simple program")
            filename = p.get("filename", f"generated_code.py")
            result = write_code(language, description, filename)

        elif intent.intent_type == "summarize":
            text = p.get("text") or prior_output or ""
            result = summarize(text)

        elif intent.intent_type == "general_chat":
            message = p.get("message", "")
            result = general_chat(message, history)

        else:
            result = ToolResult(
                intent_type="unknown",
                success=False,
                output="I didn't understand that command. Please try again with a clearer instruction.",
            )

        results.append(result)
        if result.success:
            prior_output = result.output

    return results
```

### Step 4 Verification Checklist
- [ ] No import errors: `uv run python -c "from agent.tools import execute_intents; print('OK')"`
- [ ] Smoke test — create a file:
  ```bash
  uv run python -c "
  from agent.tools import create_file
  r = create_file('test.txt', 'hello world')
  print(r)
  "
  ```
  Expected: `ToolResult(intent_type='create_file', success=True, output='File created: output/test.txt', ...)`
- [ ] Verify `output/test.txt` was created and contains `hello world`
- [ ] Path traversal blocked:
  ```bash
  uv run python -c "
  from agent.tools import create_file
  r = create_file('../../etc/passwd', 'pwned')
  print(r)
  "
  ```
  Expected: `success=False` with "Path traversal detected" message

### Step 4 STOP & COMMIT
**STOP & COMMIT:** Stop here. Run all three verification checks, then commit: `feat: tool execution — create_file, write_code, summarize, general_chat, compound pipeline`

---

## Step 5: Session Memory

- [ ] Create `agent/memory.py` with the contents below:

```python
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from .models import Intent, ToolResult

_HISTORY_FILE = Path("output/session_history.json")
_SESSION_KEY = "memo_history"


class HistoryEntry:
    """Immutable snapshot of one complete pipeline run."""

    def __init__(
        self,
        transcription: str,
        intents: list[Intent],
        results: list[ToolResult],
        audio_source: str = "unknown",
    ) -> None:
        self.timestamp = datetime.now().isoformat()
        self.audio_source = audio_source
        self.transcription = transcription
        # Store as plain dicts for JSON-safe serialization
        self.intents = [i.model_dump() for i in intents]
        self.results = [r.model_dump() for r in results]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "audio_source": self.audio_source,
            "transcription": self.transcription,
            "intents": self.intents,
            "results": self.results,
        }


class SessionMemory:
    """Manages session history backed by st.session_state with optional JSON persistence."""

    @staticmethod
    def _ensure() -> None:
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = []

    @staticmethod
    def add(entry: HistoryEntry) -> None:
        """Append a new history entry and persist to disk."""
        SessionMemory._ensure()
        st.session_state[_SESSION_KEY].append(entry.to_dict())
        SessionMemory._persist()

    @staticmethod
    def get_all() -> list[dict[str, Any]]:
        """Return all history entries in chronological order."""
        SessionMemory._ensure()
        return st.session_state[_SESSION_KEY]

    @staticmethod
    def get_chat_history() -> list[dict[str, str]]:
        """Return simplified {user, assistant} pairs for LLM context injection."""
        entries = SessionMemory.get_all()
        history: list[dict[str, str]] = []
        for entry in entries:
            user_msg = entry.get("transcription", "")
            # Collect successful outputs as the "assistant" side
            assistant_parts = [
                r["output"]
                for r in entry.get("results", [])
                if r.get("success") and r.get("output")
            ]
            if user_msg and assistant_parts:
                history.append(
                    {"user": user_msg, "assistant": " | ".join(assistant_parts)}
                )
        return history

    @staticmethod
    def clear() -> None:
        """Wipe session history from state and disk."""
        st.session_state[_SESSION_KEY] = []
        if _HISTORY_FILE.exists():
            _HISTORY_FILE.unlink()

    @staticmethod
    def _persist() -> None:
        """Write current history to JSON file (non-critical — failures are silently ignored)."""
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = SessionMemory.get_all()
            _HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
```

### Step 5 Verification Checklist
- [ ] No import errors: `uv run python -c "from agent.memory import SessionMemory, HistoryEntry; print('OK')"`
- [ ] All five modules importable together:
  ```bash
  uv run python -c "
  from agent.stt import transcribe_audio
  from agent.intent import classify_intent
  from agent.tools import execute_intents
  from agent.memory import SessionMemory, HistoryEntry
  from agent.models import Intent, ToolResult, PipelineResult
  print('All agent modules OK')
  "
  ```
  Expected: `All agent modules OK`

### Step 5 STOP & COMMIT
**STOP & COMMIT:** Stop here. Verify all imports succeed, then commit: `feat: session memory — st.session_state backed history, JSON persistence, chat context export`

---

## Step 6: Streamlit UI

- [ ] Create `app.py` in the project root with the contents below:

```python
import io

import streamlit as st

from agent.intent import classify_intent
from agent.memory import HistoryEntry, SessionMemory
from agent.stt import transcribe_audio
from agent.tools import execute_intents

st.set_page_config(
    page_title="MemO-Al Voice Agent",
    page_icon="🎙️",
    layout="wide",
)

# Persist the last pipeline result across reruns so it survives button state reset
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ---------------------------------------------------------------------------
# Sidebar — Session History
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session History")

    if st.button("🗑 Clear History", type="secondary", use_container_width=True):
        SessionMemory.clear()
        st.session_state["last_result"] = None
        st.rerun()

    history = SessionMemory.get_all()
    if not history:
        st.caption("No commands yet. Record or upload audio to get started.")
    else:
        for i, entry in enumerate(reversed(history)):
            idx = len(history) - i
            label = entry["transcription"]
            short = label[:38] + "…" if len(label) > 38 else label
            with st.expander(f"#{idx} — {short}", expanded=False):
                st.caption(
                    f"Source: {entry['audio_source']}  |  {entry['timestamp'][:19]}"
                )
                st.write(f"**Transcription:** {entry['transcription']}")
                for r in entry["results"]:
                    icon = "✅" if r["success"] else "❌"
                    intent_label = r["intent_type"].replace("_", " ").title()
                    preview = r["output"][:120] + ("…" if len(r["output"]) > 120 else "")
                    st.write(f"{icon} **{intent_label}:** {preview}")

# ---------------------------------------------------------------------------
# Main — Header
# ---------------------------------------------------------------------------
st.title("🎙️ MemO-Al — Voice-Controlled AI Agent")
st.caption(
    "Faster-Whisper (local STT, base model) · Groq LLM (llama-3.3-70b) · Apple Silicon optimized"
)

# ---------------------------------------------------------------------------
# Main — Audio Input (two tabs)
# ---------------------------------------------------------------------------
tab_mic, tab_upload = st.tabs(["🎤 Microphone", "📁 Upload File"])

audio_bytes: bytes | None = None
audio_source: str = "unknown"

with tab_mic:
    st.info(
        "Click the microphone button to record your command, then click **▶ Process Command**.",
        icon="ℹ️",
    )
    mic_audio = st.audio_input("Record your voice command")
    if mic_audio:
        audio_bytes = mic_audio.getvalue()  # non-destructive; safe to call multiple times
        audio_source = "microphone"

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a .wav or .mp3 file",
        type=["wav", "mp3"],
    )
    if uploaded:
        audio_bytes = uploaded.getvalue()
        audio_source = f"upload:{uploaded.name}"
        st.audio(uploaded)

st.divider()

# ---------------------------------------------------------------------------
# Main — Process Button
# ---------------------------------------------------------------------------
process_clicked = st.button(
    "▶ Process Command",
    type="primary",
    use_container_width=True,
    disabled=(audio_bytes is None),
)

if process_clicked and audio_bytes:
    error: str | None = None
    transcription = ""
    intents = []
    results = []

    # Step 1: Speech-to-Text
    with st.spinner("Transcribing audio (first run downloads ~145MB model)…"):
        try:
            transcription = transcribe_audio(io.BytesIO(audio_bytes))
        except Exception as exc:
            error = f"STT failed: {exc}"

    if error:
        st.error(error)
    elif not transcription.strip():
        st.warning(
            "No speech detected. Make sure the audio is not silent and try again.",
            icon="⚠️",
        )
    else:
        # Step 2: Intent Classification
        with st.spinner("Classifying intent…"):
            try:
                intents = classify_intent(
                    transcription, history=SessionMemory.get_chat_history()
                )
            except Exception as exc:
                error = f"Intent classification failed: {exc}"

        if error:
            st.error(error)
        else:
            # Step 3: Tool Execution
            with st.spinner("Executing action…"):
                try:
                    results = execute_intents(
                        intents, history=SessionMemory.get_chat_history()
                    )
                except Exception as exc:
                    error = f"Tool execution failed: {exc}"

            if error:
                st.error(error)
            else:
                # Persist result and save to session memory
                entry = HistoryEntry(transcription, intents, results, audio_source)
                SessionMemory.add(entry)
                st.session_state["last_result"] = entry.to_dict()

# ---------------------------------------------------------------------------
# Main — Results Display (persists across reruns via session_state)
# ---------------------------------------------------------------------------
if st.session_state["last_result"]:
    data = st.session_state["last_result"]

    st.subheader("📝 Transcription")
    st.info(data["transcription"])

    st.subheader("🎯 Detected Intents")
    intent_list = data.get("intents", [])
    if intent_list:
        cols = st.columns(len(intent_list))
        for col, intent in zip(cols, intent_list):
            col.metric(
                label=intent["intent_type"].replace("_", " ").title(),
                value=f"{intent.get('confidence', 1.0):.0%}",
            )

    st.subheader("⚙️ Actions & Output")
    for r in data.get("results", []):
        intent_label = r["intent_type"].replace("_", " ").title()
        if r["success"]:
            st.success(f"**{intent_label}** — completed successfully")
            if r.get("file_path"):
                st.caption(f"File saved: `{r['file_path']}`")
            if r.get("output"):
                if r["intent_type"] == "write_code":
                    st.code(r["output"])
                else:
                    output_text = r["output"]
                    height = min(300, max(80, output_text.count("\n") * 22 + 60))
                    st.text_area(
                        "output",
                        output_text,
                        height=height,
                        disabled=True,
                        label_visibility="collapsed",
                    )
        else:
            st.error(f"**{intent_label}** — {r['output']}")
```

### Step 6 Verification Checklist
- [ ] App launches without errors:
  ```bash
  uv run streamlit run app.py
  ```
  Expected: browser opens at `http://localhost:8501` with title "MemO-Al Voice Agent"
- [ ] **Microphone tab** visible with record button
- [ ] **Upload File tab** visible with file uploader
- [ ] **Process Command** button is disabled when no audio is provided
- [ ] **Sidebar** shows "No commands yet." initially
- [ ] Upload a `.wav` file → button becomes enabled → click Process → transcription appears
- [ ] Say/type "Create a file called hello.txt with content: hello world" (as audio) → `output/hello.txt` created → UI shows success
- [ ] Try garbled/silent audio → warning banner displayed, no crash
- [ ] Run two commands → sidebar shows both entries with expandable details
- [ ] Click "Clear History" → sidebar resets, last result cleared

### Step 6 STOP & COMMIT
**STOP & COMMIT:** Stop here. Run all UI verification checks, then commit: `feat: Streamlit UI — mic/upload tabs, pipeline display, session history sidebar, graceful errors`

---

## Step 7: README & Polish

- [ ] Replace the contents of `README.md` with the contents below:

```markdown
# MemO-Al — Voice-Controlled Local AI Agent

A voice-controlled AI agent that transcribes speech locally, classifies intent via LLM, executes tools, and displays the full pipeline in a Streamlit UI.

## Architecture

```
Audio Input (mic or file upload)
        ↓
STT — Faster-Whisper (local, base model, ~145 MB, offline)
        ↓
Intent Classification — Groq API (llama-3.3-70b-versatile, JSON Object Mode)
        ↓
Tool Execution — create_file | write_code | summarize | general_chat
        ↓
Session Memory — st.session_state + output/session_history.json
        ↓
Streamlit UI — transcription · intents · actions · output
```

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| STT | Faster-Whisper (local, base) | Free, offline, no GPU needed, Apple Silicon compatible |
| LLM | Groq API — llama-3.3-70b-versatile | Fast inference, free tier, JSON Object Mode support |
| UI | Streamlit ≥1.32 | `st.audio_input()` for in-browser mic recording |
| Models | Pydantic v2 | Structured intent/result schemas |
| Runtime | Python 3.12, uv | Fast dependency resolution |

## Hardware Note — Apple Silicon

Faster-Whisper uses CTranslate2, which does not support Apple's Metal (MPS) backend.
The agent uses `device="cpu"` with `compute_type="int8"` quantization.
On M-series chips (M1/M2/M3), the base model transcribes a 10-second clip in under 2 seconds.

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- A free [Groq API key](https://console.groq.com) (free tier: 1,000 req/day)

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your GROQ_API_KEY
```

### 3. Run

```bash
uv run streamlit run app.py
```

The first launch downloads the Whisper `base` model (~145 MB, ~30–60 seconds).
The model is cached locally and all subsequent runs are instant.

## Usage

1. Open `http://localhost:8501` in your browser
2. Choose the **Microphone** tab to record a command (click the mic button, speak, click stop)
3. Or choose the **Upload File** tab to upload a `.wav` or `.mp3`
4. Click **▶ Process Command**
5. View the transcription, detected intents, and output in the main panel
6. Review command history in the left sidebar

## Supported Commands

| Intent | Example Utterance |
|---|---|
| create_file | "Create a file called notes.txt with the content hello world" |
| write_code | "Write a Python function that reverses a string and save it to reverse.py" |
| summarize | "Summarize this: The quick brown fox jumps over the lazy dog. It did so repeatedly." |
| general_chat | "What is the difference between a list and a tuple in Python?" |
| **Compound** | "Summarize this text and save the result to summary.txt" |

## Bonus Features

- **Compound Commands** — Multiple intents in a single utterance. The output of one step (e.g., a summary) is automatically piped as input to the next (e.g., create a file).
- **Graceful Degradation** — Silent audio, unintelligible speech, unknown intents, and API errors all display user-friendly banners without crashing.
- **Session Memory** — Full command history in the sidebar. Entries persist within the session and are saved to `output/session_history.json`.

## Output Safety

All file operations are restricted to the `output/` directory.
Path traversal attempts (e.g., `../../etc/passwd`) are blocked via `pathlib.Path.is_relative_to()` checks.

## Project Structure

```
app.py                        ← Streamlit entry point
agent/
    __init__.py
    stt.py                    ← Faster-Whisper transcription wrapper
    intent.py                 ← Groq intent classifier (JSON Object Mode)
    tools.py                  ← Tool execution (create_file, write_code, summarize, chat)
    memory.py                 ← Session history (st.session_state + JSON file)
    models.py                 ← Pydantic v2 schemas (Intent, ToolResult, PipelineResult)
output/                       ← All file operations land here (git-ignored contents)
.env                          ← API keys (not committed)
.env.example                  ← Template
pyproject.toml
```
```

### Step 7 Verification Checklist
- [ ] `README.md` renders correctly (check in GitHub preview or VS Code Markdown preview)
- [ ] All setup instructions produce a working app when followed from scratch
- [ ] Architecture diagram is accurate and matches the actual code

### Step 7 STOP & COMMIT
**STOP & COMMIT:** Stop here. Review the README, then commit: `docs: README — architecture, setup, usage, hardware note, bonus features`

---

## Final Verification

Run through the complete end-to-end checklist before submitting:

- [ ] `uv sync` — no errors
- [ ] `uv run streamlit run app.py` — UI loads at `http://localhost:8501`
- [ ] Upload a `.wav` file → transcription appears
- [ ] Command: "Create a file called hello.txt with the text: Hello from MemO-Al" → `output/hello.txt` created, UI shows success
- [ ] Command: "Write a Python function that calculates factorial and save it to factorial.py" → `output/factorial.py` created with working code
- [ ] Command: "Summarize this text and save it to summary.txt: Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction." → two intents fire in sequence → `output/summary.txt` created with summary content
- [ ] Silent/garbled audio → warning banner, no crash, no stack trace
- [ ] Run 3+ commands → sidebar shows all entries with correct expandable details
- [ ] Click "Clear History" → sidebar empties, `output/session_history.json` deleted
- [ ] `.env` not tracked by git (`git status` shows it clean or untracked-ignored)
- [ ] `output/` contents ignored by git (only `.gitkeep` tracked)
