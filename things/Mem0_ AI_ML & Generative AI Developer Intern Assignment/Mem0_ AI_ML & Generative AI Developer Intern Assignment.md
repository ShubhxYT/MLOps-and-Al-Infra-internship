## Assignment: Build a Voice-Controlled Local AI Agent

## Objective

Your goal is to build a local AI agent that can accept audio input, accurately classify the user's intent, execute the appropriate local tools, and display the entire pipeline's results in a clean User Interface (UI).

## System Requirements

## 1. Audio Input

The system must be able to accept audio via two methods:

- Direct microphone input.
- ●
- Uploading an existing audio file (e.g., .wav or .mp3).

## 2. Speech-to-Text (STT)

Convert the provided audio into text.

- Requirement: Use a HuggingFace model (such as Whisper or wav2vec) or another local model.
- Hardware Note: If your local machine cannot run this model efficiently, you may use an API-based STT service (like Groq or OpenAI). If you choose this route, please document why in your README.

## 3. Intent Understanding

Use a Large Language Model (local model preferred via Ollama, LM Studio, etc.) to analyze the transcribed text and classify the user's intent.

- Supported Intents (Minimum):
- Create a file.
- Write code to a new or existing file.
- Summarize text.
- General Chat.

## 4. Tool Execution

Based on the detected intent, the system should trigger specific actions on your local machine.

- File Operations: Create files or folders.

○

Safety Constraint:

Please restrict all file creation and code writing to a dedicated output/ folder within your repository to avoid accidental system overwrites.

- Code Generation: Generate code and save it directly into a file.
- Text Processing: Summarize provided content.

## 5. User Interface

Build a frontend to interact with your agent using Streamlit, Gradio, or another Web UI framework of your choice.

## ● The UI must display:

- The transcribed text from the audio.
- The detected intent.
- The specific action taken by the system.
- The final output or result.

## Example Flow

User says: "Create a Python file with a retry function."

## System Execution:

1. Transcribes audio to text.
2. Detects intent: "Write code" and "Create file".
3. Generates the Python code.
4. Creates the file locally in the designated output folder.
5. Shows the transcription, intent, and success message in the UI.

## Deliverables

1. Code Repository: A public GitHub link containing your code. It must include a README.md with clear setup instructions, architecture explanation, and any hardware workarounds used.
2. Video Demo: A short, 2-3 minute video (YouTube Unlisted link) demonstrating the working UI and system flow. Show at least two different intents working successfully.
3. Technical Article: Publish a short article on Substack, Dev.to, or Medium explaining your architecture, the models you chose, and the challenges you faced while building this agent. Provide the link.

## Submission Instructions

Once you have completed the code, recorded your video, and published your article, please submit all links via this official form: Submit Your Project Here https://forms.gle/5x32P7zr4NvyRgK6A

Note: Late submissions will not be accepted.

## Bonus (Optional)

Want to stand out? Consider implementing one or more of the following:

- Compound Commands: Support multiple commands in one audio input (e.g., "Summarize this text and save it to summary.txt" ).
- Human-in-the-Loop: Add a UI confirmation prompt before executing file operations.
- Graceful Degradation: Handle errors smoothly (e.g., unintelligible audio, unmapped intents).
- Memory: Maintain a persistent history of actions and chat context within the session.
- Model Benchmarking: Include a section in your article comparing the performance/speed of your chosen models.