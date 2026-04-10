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
