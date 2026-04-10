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
