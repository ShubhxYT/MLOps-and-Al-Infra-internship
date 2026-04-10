# Voice-Controlled Local AI Agent (MemO-Al assignment)

**Description:** Build a full-stack voice-controlled AI agent with STT → intent classification → tool execution → Streamlit UI.

## User Decisions
- Machine: Apple Silicon (M-series), Python 3.12, package manager: uv
- STT: Faster-Whisper (local, Apple Silicon MPS-accelerated)
- LLM: Groq API (llama3/mixtral)
- UI: Streamlit
- Bonus: compound commands, graceful degradation, session memory

## Project Structure (final)
```
app.py                    ← Streamlit entry point
agent/
  __init__.py
  stt.py                  ← Faster-Whisper (local) wrapper
  intent.py               ← Groq intent classifier (structured JSON)
  tools.py                ← Tool execution (create_file, write_code, summarize, chat)
  memory.py               ← Session history (state + JSON file)
  models.py               ← Pydantic schemas (Intent, ToolResult, etc.)
output/                   ← SAFE folder for all file ops (git-ignored content)
.env                      ← API keys (not committed)
.env.example
pyproject.toml            ← updated deps
```

## Implementation Steps

### Commit 1: Project Setup
Files: pyproject.toml, .env.example, .gitignore, agent/__init__.py, output/.gitkeep
- Add deps: faster-whisper, groq, streamlit, python-dotenv, pydantic
- Create agent/ package and output/ folder
- .env.example with GROQ_API_KEY (no OpenAI key needed)

### Commit 2: STT Module
Files: agent/stt.py
- Faster-Whisper local model (base or small, ~500MB–1GB download)
- Device detection: auto-use Apple Silicon MPS if available, fallback to CPU
- Input: file path or BytesIO (from mic or file upload)
- Output: str transcription
- Errors: empty audio, model download failure → raise with descriptive message

### Commit 3: Intent Classification
Files: agent/models.py, agent/intent.py
- Pydantic models: Intent(intent_type, params), PipelineResult
- Intent types: create_file, write_code, summarize, general_chat, unknown
- Groq call with system prompt that returns JSON list of intents (enables compound)
- Confidence field; falls back to unknown on low confidence or parse failure

### Commit 4: Tool Execution
Files: agent/tools.py
- create_file(filename, content): writes to output/filename, path sanitization
- write_code(language, description, filename): Groq codegen → output/filename
- summarize(text): Groq summarization
- general_chat(message, history): Groq conversational with history
- Compound execution: runs intents in sequence, passes prior output to next intent if applicable
- Safety: resolve all paths under output/, strip ../ traversal

### Commit 5: Session Memory
Files: agent/memory.py
- SessionMemory class: wraps list of HistoryEntry (timestamp, audio_source, transcription, intents, results)
- Backed by st.session_state["memo_history"]
- Optional JSON persistence to output/session_history.json
- Methods: add(), get_all(), clear()

### Commit 6: Streamlit UI
Files: app.py
- Layout:
  - Sidebar: session history (expandable entries), Clear History button
  - Main: title, two tabs (Mic | Upload File)
  - st.audio_input() for mic (Streamlit ≥1.32)
  - st.file_uploader() for .wav/.mp3
  - "Process" button → runs full pipeline
  - Results section: Transcription box, Intent pills, Action taken, Output (code block or text)
  - Error banner for graceful degradation (unintelligible audio, unknown intent, API errors)

### Commit 7: README & Polish
Files: README.md
- Setup instructions (uv sync, .env, streamlit run app.py)
- Architecture diagram (text)
- Hardware note: using Faster-Whisper for zero cost, Apple Silicon MPS acceleration; Groq API for fast intent classification (free tier or low-cost API)
- First-run model download (~30–60s): base model cached locally
- Bonus features documented

## Verification
1. `uv sync` installs all deps without errors
2. `streamlit run app.py` launches UI on localhost
3. Upload a .wav file → transcription appears in UI
4. Say "create a file called hello.txt" → output/hello.txt created, UI shows result
5. Say "summarize this text and save it to summary.txt" → compound intent fires both summarize + create_file
6. Give garbled/silent audio → graceful error shown (no crash)
7. Session history persists across multiple voice commands in same session

## Decisions / Scope
- All file ops restricted to output/ (safety constraint from assignment)
- STT: Local Faster-Whisper with Apple Silicon MPS acceleration — zero cost, offline-capable
- LLM: Groq API (not local Ollama) — faster cold start, no local model management needed
- mem0ai library NOT included (adds infra complexity; session_state + JSON file is sufficient for "session memory" bonus)
- No HITL confirmation prompt (not selected by user)
- dockling.py left untouched (PDF tool, separate concern)
