# Cornerlius — Phases 1–4: foundation, memory, safe tools, agent loop

A modular, local-first Python desktop assistant for Windows. It includes a modern CustomTkinter chat window, SQLite conversation history, explicit searchable long-term memory, a provider interface, an offline mock model, an optional Ollama local chat adapter, and a safe audited tool registry. No API key is required. (The package and `JARVIS_*` setting names are unchanged for compatibility.)

## Requirements

- Windows 10/11 and Python 3.10 or newer (3.12 recommended).
- Interface dependencies: `pip install -r jarvis/requirements.txt` (CustomTkinter and PySide6).
- Optional: Ollama installed separately, with a model downloaded locally. The GTX 1050 Ti's 4 GB VRAM is best suited to small quantized models; `qwen2.5:3b` is the configurable default, but actual speed and memory use depend on quantization and context length. CPU inference may work with reduced speed. Cornerlius does not download or claim a model is available.

## Run

Two interfaces share the same backend. The Qt interface is the current one:

```powershell
python -m jarvis.main_qt
```

The older CustomTkinter interface remains available until the Qt cutover is approved:

```powershell
python -m jarvis.main
```

Mock mode is the default and works offline. Chat messages are stored under `%USERPROFILE%\.jarvis\jarvis.sqlite3` unless `JARVIS_DATA_DIR` is set. The authorized file workspace defaults to `%USERPROFILE%\jarvis-workspace` unless `JARVIS_WORKSPACE` is set.

To use a running local Ollama service, configure environment variables before launch:

```powershell
$env:JARVIS_MODEL_PROVIDER = "ollama"
$env:JARVIS_OLLAMA_MODEL = "qwen2.5:3b"
python -m jarvis.main
```

The adapter calls Ollama's local `/api/chat` endpoint. If it is unavailable or the configured model is missing, Cornerlius displays the connection error and retains the user's message in conversation history.

## Configuration

`JARVIS_MODEL_PROVIDER` is `mock` or `ollama`; `JARVIS_DATA_DIR`, `JARVIS_OLLAMA_URL`, `JARVIS_OLLAMA_MODEL`, `JARVIS_REQUEST_TIMEOUT`, `JARVIS_WORKSPACE`, `JARVIS_TOOL_TIMEOUT`, `JARVIS_AGENT_MAX_ITERATIONS`, `JARVIS_AGENT_TIMEOUT`, and `JARVIS_AGENT_RETRY` can also be configured. `.env.example` documents the settings; the app does not parse `.env` files automatically.

### Storage layout (this machine)

Everything lives on `D:` to spare the system drive:

- Ollama app and models: `D:\Ollama` (`OLLAMA_MODELS=D:\Ollama\models`)
- Cornerlius database: `D:\Jarvis\data` (`JARVIS_DATA_DIR`)
- Cornerlius workspace: `D:\Jarvis\workspace` (`JARVIS_WORKSPACE`)

These are saved as user environment variables, so every launch uses them automatically.

## Verify

```powershell
python -m unittest discover -s jarvis/tests -v
```

The tests use temporary SQLite databases, a temporary workspace, and the offline model. Launching the UI requires an interactive Windows desktop session.

## Just talk — no commands needed

Every message goes through the normal chat path first. Plain conversation is answered directly, with replies **streaming word-by-word** as the model generates them; when the model decides a tool is needed, the bounded agent loop runs it automatically, so these all work as written. In the desktop app, **Enter sends** and Shift+Enter adds a new line.

```text
what does my todo say?
write hello.txt containing hi
remember that my cat is named Pixel
remember that I prefer short answers
what do you remember about my cat?
forget my cat
```

Writes pause before changing anything: Cornerlius tells you what it wants to do and you reply **yes** or **no**. While paused you can keep chatting; a new request drops the paused run with a notice (nothing was changed yet, so this is safe). All `/`-commands (`/agent`, `/tool`, `/remember`, `/memory`, `/forget`, `/confirm`, `/cancel`, `/audit`) keep working as explicit overrides. Ordinary statements are never turned into permanent memory — only explicit "remember ..." phrasing saves.

## Speed notes

- Chat replies stream token-by-token (`submit_stream` / `chat.send_stream`), so reading starts with the first tokens instead of after the full reply.
- The Ollama adapter sends `keep_alive: 30m`, keeping model weights in VRAM between chats so follow-up replies start fast.
- Python orchestration costs milliseconds per message; inference dominates, so there is no native-accelerated orchestration layer — Ollama itself (llama.cpp/CUDA) is already the optimized native component.
- If replies feel slow, prefer a smaller local model or shorter answers over a bigger download: model size dominates speed on a 4 GB GPU.

## Phase 1 scope

The application persists a default conversation and passes recent messages plus matching saved memory to the selected provider. Tk widgets are only touched on the UI thread; model requests run in a background worker.

## Long-term memory

Cornerlius only saves long-term memory when you explicitly ask. Enter these in the chat box:

```text
/remember My cat is named Pixel
/remember preference | I prefer concise answers
/remember project | Cornerlius uses SQLite for local memory
/memory
/memory search Pixel
/memory edit 2 | I prefer short bullet points
/forget 2
```

Categories are `fact` (the default), `preference`, and `project`. `/memory` lists saved entries with their IDs. Search, edit, and deletion are explicit; all entries are stored locally in the same SQLite database. Matching entries are added as reference context for configured model providers. The mock provider remains deterministic and does not generate memory-aware answers; use `/memory search ...` to retrieve entries in mock mode.

## Safe tools (Phase 3)

Tools run through `jarvis/tools/registry.py` with argument schemas, type validation, per-call timeouts, structured success/failure results, and an append-only `tool_audit` table in the same SQLite database. Audit entries store truncated argument summaries, never full file contents. There is intentionally no shell tool.

Available tools:

- `workspace.list` (read-only): list files inside the workspace.
- `workspace.read` (read-only): read a text file inside the workspace.
- `workspace.write` (confirmation-required): create or overwrite a text file inside the workspace.
- `system.info` (read-only): OS, CPU count, and Python version.
- `web.search` (confirmation-required): search the web via DuckDuckGo, no API key. Queries leave this machine, so approval is required.
- `web.fetch` (confirmation-required): fetch a public http(s) page as plain text. Local/private hosts, non-http schemes, and non-text content are refused.

Every path is resolved and rejected when it escapes the workspace root. Reads are size-limited and writes are capped at 500,000 characters. A write first returns a confirmation prompt; nothing is written until you reply **yes**. Reply **no** to abort.

```text
/tools
/tool system.info {}
/tool workspace.list {"path": ".", "recursive": false}
/tool workspace.read {"path": "notes/todo.txt"}
/tool workspace.write {"path": "notes/todo.txt", "content": "buy milk", "create_dirs": true}
/confirm
/cancel
/audit 10
```

Tool results are inspected before reporting: failures are reported as failures, never as success. Voice, task recovery, and broader automation remain later phases.

## Bounded agent loop (Phase 4)

Multi-step goals run through `jarvis/core/agent.py` with hard budgets: at most `JARVIS_AGENT_MAX_ITERATIONS` model turns (default 6), an overall `JARVIS_AGENT_TIMEOUT` (default 120s), per-tool timeouts, and one retry for transient timeouts only — validation and permission errors are never retried. The run stops with a partial step list on cancellation, timeout, model errors, or the iteration cap, so it cannot loop forever.

```text
/agent --allow-write Summarise notes/todo.txt into notes/summary.txt
/agent --allow-web What year was the first Amazing Spider-Man movie?
/agent Draft release notes from the workspace changelog
/confirm
/cancel
```

The model requests tools with a fenced block (```tool {"tool": ..., "args": {...}}```); anything else is treated as the final answer. Writes still need approval unless the run was started with `--allow-write`; otherwise the run pauses and `/confirm` resumes it where it stopped. Every write is verified by re-reading the file, and the final summary marks each step ok/failed plus verified, with a warning when steps failed. The desktop shows live progress per step in the status bar and has a Cancel button that stops the run after its current step. (Cancellation is cooperative: an in-flight local-model HTTP call finishes first.)

## Teaching Cornerlius (dynamic interaction and training)

Cornerlius does not retrain model weights — on 4 GB VRAM that is not realistic. Instead it learns the way an assistant should: you teach it, it remembers, and it applies the lesson. Anything saved with `/remember` (facts, preferences, project notes) is retrieved as reference context for both chat and agent runs:

```text
/remember preference | Always keep summaries under five bullets
/agent --allow-write Summarise notes/todo.txt
```

The preference above is injected into the agent's instructions, so the run follows it. Correct it anytime with `/memory edit`, remove it with `/forget`, and inspect what ran with `/audit`. While an agent run is paused for confirmation you can keep chatting normally, then `/confirm` to resume or `/cancel` to abort.

## Persistent tasks and reusable skills (Day 3)

Every agent run is recorded in SQLite (`task_runs`), so a paused or interrupted task survives restarts:

```text
/agent --allow-write Summarise notes/todo.txt into notes/summary.txt
/tasks
/resume 3
```

Successful runs can become versioned skills. A skill is saved as a draft, must pass validation before it can be enabled, and is never executed automatically:

```text
/skill save summarise-notes
/skills
/skill test summarise-notes
/skill enable summarise-notes
```

Memory edits are logged as corrections (`memory_corrections`) so later improvements can prefer corrected content. Skill execution, PC automation (screenshots/OCR/input), and a central permission manager remain future work.
