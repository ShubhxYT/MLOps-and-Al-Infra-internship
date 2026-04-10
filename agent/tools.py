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
