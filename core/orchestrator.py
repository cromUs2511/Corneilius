from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jarvis.config import Settings
from jarvis.core.agent import (
    AgentStep,
    build_agent_system,
    build_chat_system,
    parse_tool_request,
    summarize,
    truncate_feed,
    verify_write,
)
from jarvis.core.natural import (
    guess_category,
    is_affirmative,
    is_negative,
    parse_forget,
    parse_recall,
    parse_remember,
)
from jarvis.memory.database import ConversationStore
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.skills import SkillStore
from jarvis.memory.tasks import TaskRunStore
from jarvis.models.base import ModelAdapter
from jarvis.tools.audit import ToolAuditLog
from jarvis.tools.registry import ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)

AGENT_USAGE = "Usage: /agent [--allow-write] [--allow-web] [--verbose] <goal>"


class Orchestrator:
    def __init__(
        self,
        model: ModelAdapter,
        memory: ConversationStore,
        settings: Settings | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.model = model
        self.memory = memory
        self.long_term = LongTermMemory(memory.path)
        self.tasks = TaskRunStore(memory.path)
        self.skills = SkillStore(memory.path)
        self._pending: dict[str, Any] | None = None
        self._pending_agent: dict[str, Any] | None = None
        self._agent_cancel = threading.Event()
        self._last_final = ""
        self._last_replaced = False
        self._last_task_id: int | None = None
        self._last_steps: list[AgentStep] = []
        self._last_goal = ""
        # Optional live-progress hook (used by the desktop UI to show each step).
        # Called from a worker thread; implementations must be thread-safe.
        self.on_progress: Any = None
        if settings is None:
            self._agent_max_iterations = 6
            self._agent_timeout = 120.0
            self._agent_retry = 1
        else:
            self._agent_max_iterations = max(1, min(int(settings.agent_max_iterations), 20))
            self._agent_timeout = max(5.0, float(settings.agent_timeout))
            self._agent_retry = max(0, min(int(settings.agent_retry), 3))
        if tools is not None:
            self.tools = tools
        else:
            if settings is None:
                # Keep tests and ad-hoc use isolated: default the workspace
                # next to the database instead of the real home directory.
                workspace = memory.path.parent / "workspace"
                timeout = 10.0
            else:
                workspace = settings.workspace_root
                timeout = settings.tool_timeout
            self.tools = build_default_registry(ToolAuditLog(memory.path), Path(workspace), timeout)

    def cancel_agent(self) -> None:
        """Request cooperative cancellation of a running agent loop."""
        self._agent_cancel.set()

    def _report_progress(self, text: str) -> None:
        hook = self.on_progress
        if hook is None:
            return
        try:
            hook(text)
        except Exception:
            logger.warning("Progress hook failed", exc_info=True)

    def submit(self, message: str) -> str:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        command = cleaned.partition(" ")[0].casefold()
        if command not in {"/cancel", "/cancel-agent", "/confirm"}:
            self._agent_cancel.clear()
        self.memory.add_message("user", cleaned)
        try:
            answer, prefix = self._route_sync(cleaned)
            if answer is None:
                answer = prefix + self._chat(cleaned)
            if not answer.strip():
                raise RuntimeError("The model returned an empty response.")
            self.memory.add_message("assistant", answer)
            return answer
        except Exception:
            logger.exception("Assistant response failed")
            raise

    def _route_sync(self, cleaned: str) -> tuple[str | None, str]:
        """Run every non-chat path. Returns (answer, prefix); answer None means chat."""
        answer = self._memory_command(cleaned)
        if answer is None:
            answer = self._tool_command(cleaned)
        if answer is None:
            answer = self._agent_command(cleaned)
        if answer is None:
            answer = self._task_command(cleaned)
        if answer is None:
            answer = self._skill_command(cleaned)
        if answer is None:
            answer = self._pending_answer(cleaned)
        if answer is None:
            answer = self._natural_memory(cleaned)
            if answer is not None and self._pending_agent is not None:
                goal = str(self._pending_agent.get("goal", ""))[:60]
                answer += (
                    f"\n\nNote: the earlier request ('{goal}') is still paused — "
                    "reply yes to continue it or no to drop it."
                )
        if answer is not None:
            return answer, ""
        prefix = ""
        if self._pending_agent is not None:
            goal = str(self._pending_agent.get("goal", ""))[:60]
            self._pending_agent = None
            prefix = f"Note: I dropped the paused request ('{goal}') to handle this.\n\n"
        return None, prefix

    def submit_stream(self, message: str) -> Iterator[str]:
        """Streaming twin of submit(): yields reply chunks as they arrive.

        After exhaustion, `self._last_final` holds the authoritative reply text
        and `self._last_replaced` tells whether the streamed preview must be
        discarded (tool-escalation path) rather than kept.
        """
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        command = cleaned.partition(" ")[0].casefold()
        if command not in {"/cancel", "/cancel-agent", "/confirm"}:
            self._agent_cancel.clear()
        self.memory.add_message("user", cleaned)
        self._last_final = ""
        self._last_replaced = False
        try:
            answer, _prefix = self._route_sync(cleaned)
            if answer is not None:
                self._last_final = answer
                self.memory.add_message("assistant", answer)
                yield answer
                return
            history = self._chat_history(cleaned)
            stream = self.model.stream_respond(cleaned, history)
            parts: list[str] = []
            try:
                for chunk in stream:
                    if self._agent_cancel.is_set():
                        break
                    if chunk:
                        parts.append(chunk)
                        yield chunk
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            full = "".join(parts)
            if not full.strip():
                if self._agent_cancel.is_set():
                    return  # cancelled before anything arrived: nothing to store
                raise RuntimeError("The model returned an empty response.")
            try:
                request = parse_tool_request(full)
                malformed = False
            except ValueError:
                # Malformed tool block: let the loop record it so the model can self-correct.
                request, malformed = None, True
            if request is None and not malformed:
                self._last_final = full
                self.memory.add_message("assistant", full)
                return
            summary = self._agent_run(cleaned, False, first_reply=full)
            self.memory.add_message("assistant", summary)
            self._last_final = summary
            self._last_replaced = True
            yield summary
        except Exception:
            logger.exception("Assistant streaming response failed")
            raise

    def _memory_command(self, message: str) -> str | None:
        """Handle explicit memory actions through the normal chat input."""
        command, _, argument = message.partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command == "/remember":
            category, separator, content = argument.partition("|")
            if separator:
                category, content = category.strip().casefold(), content.strip()
            else:
                category, content = "fact", argument
            entry_id = self.long_term.save(category, content)
            return f"Saved memory #{entry_id} ({category})."
        if command == "/memory":
            if argument.casefold().startswith("edit "):
                return self._edit_memory(argument[5:].strip())
            if argument.casefold().startswith("search "):
                query = argument[7:].strip()
                entries = self.long_term.search(query)
                return self._format_memories(entries, f"No saved memories matched ‘{query}’.")
            entries = self.long_term.list_entries()
            return self._format_memories(entries, "No saved memories yet. Use /remember to save one.")
        if command == "/forget":
            try:
                entry_id = int(argument)
            except ValueError as exc:
                raise ValueError("Usage: /forget <memory ID>") from exc
            if not self.long_term.delete(entry_id):
                return f"No saved memory with ID {entry_id}."
            return f"Deleted memory #{entry_id}."
        if command == "/memory-edit":
            return self._edit_memory(argument)
        return None

    def _edit_memory(self, argument: str) -> str:
        raw_id, separator, content = argument.partition("|")
        if not separator:
            raise ValueError("Usage: /memory edit <memory ID> | <new text>")
        try:
            entry_id = int(raw_id.strip())
        except ValueError as exc:
            raise ValueError("Memory ID must be a number.") from exc
        self.long_term.update(entry_id, content)
        return f"Updated memory #{entry_id}."

    @staticmethod
    def _format_memories(entries: list[dict[str, int | str]], empty: str) -> str:
        if not entries:
            return empty
        return "Saved memories:\n" + "\n".join(
            f"#{entry['id']} [{entry['category']}] {entry['content']}" for entry in entries
        )

    # -- Phase 3: safe tools -------------------------------------------------

    def _tool_command(self, message: str) -> str | None:
        command, _, argument = message.partition(" ")
        command = command.casefold()
        argument = argument.strip()

        if command == "/tools":
            lines = [spec.usage() for spec in self.tools.specs()]
            return "Available tools:\n" + "\n".join(lines) if lines else "No tools registered."

        if command == "/tool":
            if not argument:
                raise ValueError("Usage: /tool <name> {JSON arguments}")
            name, _, raw_args = argument.partition(" ")
            name = name.strip()
            raw_args = raw_args.strip()
            args: dict[str, Any] = {}
            if raw_args:
                try:
                    parsed = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Arguments must be a JSON object: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Arguments must be a JSON object.")
                args = parsed
            result = self.tools.execute(name, args)
            if result.confirmation_required:
                self._pending = {"name": name, "args": args}
                return (
                    f"Tool '{name}' needs confirmation because it can change files. "
                    "Reply yes to proceed or no to abort."
                )
            if result.ok:
                return f"Tool '{result.name}' succeeded in {result.elapsed_ms}ms:\n{result.output}"
            return f"Tool '{result.name}' failed: {result.error}"

        if command == "/confirm":
            if self._pending_agent is not None:
                pending = self._pending_agent
                self._pending_agent = None
                return self._agent_run(
                    str(pending["goal"]),
                    bool(pending["allow_write"]),
                    resume_steps=list(pending["steps"]),
                    resume_confirmed={"name": str(pending["step"]["tool"]), "args": dict(pending["step"]["args"])},
                    allow_web=bool(pending.get("allow_web", False)),
                    verbose=bool(pending.get("verbose", False)),
                    task_id=pending.get("task_id"),
                )
            if self._pending is None:
                return "Nothing is waiting for confirmation."
            pending = self._pending
            self._pending = None
            result = self.tools.execute(str(pending["name"]), dict(pending["args"]), confirmed=True)
            if result.ok:
                return f"Tool '{result.name}' succeeded in {result.elapsed_ms}ms:\n{result.output}"
            return f"Tool '{result.name}' failed: {result.error}"

        if command in {"/cancel", "/cancel-agent"}:
            cancelled_something = False
            self._agent_cancel.set()
            if self._pending_agent is not None:
                dropped_task = self._pending_agent.get("task_id")
                if dropped_task is not None:
                    try:
                        self.tasks.finish(int(dropped_task), "cancelled", [], "Dropped while paused.")
                    except Exception:
                        logger.warning("Could not mark task cancelled", exc_info=True)
                self._pending_agent = None
                cancelled_something = True
            if self._pending is not None:
                self._pending = None
                cancelled_something = True
            if command == "/cancel-agent" or cancelled_something:
                return "Pending tool action cancelled."
            # Plain /cancel with nothing pending still signals a running loop.
            return "Cancellation requested. A running agent will stop after its current step."

        if command == "/audit":
            limit = 10
            if argument:
                try:
                    limit = int(argument)
                except ValueError as exc:
                    raise ValueError("Usage: /audit [limit]") from exc
            # The audit log shares the conversation database file.
            entries = ToolAuditLog(self.memory.path).recent(limit)
            if not entries:
                return "No tool actions recorded yet."
            lines = [
                f"#{e['id']} {e['created_at']} {e['tool']} "
                f"[{'ok' if e['ok'] else 'fail'}] {e['error'] or e['args']}"
                for e in entries
            ]
            return "Recent tool actions:\n" + "\n".join(lines)

        return None

    def _pending_answer(self, message: str) -> str | None:
        """Plain-word yes/no handling while an agent run is paused for confirmation."""
        if self._pending_agent is None:
            return None
        if is_affirmative(message):
            pending = self._pending_agent
            self._pending_agent = None
            return self._agent_run(
                str(pending["goal"]),
                bool(pending["allow_write"]),
                resume_steps=list(pending["steps"]),
                resume_confirmed={"name": str(pending["step"]["tool"]), "args": dict(pending["step"]["args"])},
                allow_web=bool(pending.get("allow_web", False)),
                verbose=bool(pending.get("verbose", False)),
                task_id=pending.get("task_id"),
            )
        if is_negative(message):
            dropped_task = self._pending_agent.get("task_id")
            if dropped_task is not None:
                try:
                    self.tasks.finish(int(dropped_task), "cancelled", [], "Dropped while paused.")
                except Exception:
                    logger.warning("Could not mark task cancelled", exc_info=True)
            self._pending_agent = None
            return "Okay, I dropped that request. Nothing was changed."
        return None

    # -- Natural conversation (no commands needed) ---------------------------

    def _natural_memory(self, message: str) -> str | None:
        """Handle plain-language remember / recall / forget requests."""
        remembered = parse_remember(message)
        if remembered is not None:
            category = guess_category(remembered)
            entry_id = self.long_term.save(category, remembered)
            return f"Noted — saved as {category} #{entry_id}: {remembered}"

        recall = parse_recall(message)
        if recall is not None:
            kind, query = recall
            if kind == "all":
                entries = self.long_term.list_entries()
                return self._format_memories(entries, "I don't have anything saved yet.")
            entries = self.long_term.search(query or "")
            return self._format_memories(entries, f"I don't have anything saved about '{query}'.")

        forget_query = parse_forget(message)
        if forget_query is not None:
            matches = self.long_term.search(forget_query)
            if not matches:
                return f"I couldn't find anything saved about '{forget_query}', so nothing was deleted."
            if len(matches) == 1:
                self.long_term.delete(int(matches[0]["id"]))
                return f"Forgot #{matches[0]['id']}: {matches[0]['content']}"
            lines = "\n".join(
                f"#{entry['id']} [{entry['category']}] {entry['content']}" for entry in matches
            )
            return (
                "I found several matches, so I deleted nothing. "
                f"Use /forget <ID> for the exact one:\n{lines}"
            )
        return None

    def _chat_history(self, message: str) -> list[dict[str, str]]:
        seed = self.memory.get_messages(limit=12)[:-1]
        relevant = self.long_term.retrieve_relevant(message)
        memories = [f"[{entry['category']}] {entry['content']}" for entry in relevant]
        system = build_chat_system(memories, [spec.usage() for spec in self.tools.specs()])
        history: list[dict[str, str]] = [{"role": "system", "content": system}]
        history.extend({"role": row["role"], "content": row["content"]} for row in seed)
        return history

    def _chat(self, message: str) -> str:
        """Default path: one model call; escalate to the agent loop only on tool use."""
        history = self._chat_history(message)
        reply = self.model.respond(message, history)
        try:
            request = parse_tool_request(reply)
        except ValueError:
            # Malformed tool block: let the loop record it so the model can self-correct.
            return self._agent_run(message, False, first_reply=reply)
        if request is None:
            return reply
        return self._agent_run(message, False, first_reply=reply)

    # -- Phase 4: bounded agent loop -----------------------------------------

    def _agent_command(self, message: str) -> str | None:
        command, _, argument = message.partition(" ")
        if command.casefold() != "/agent":
            return None
        argument = argument.strip()
        allow_write = False
        allow_web = False
        verbose = False
        while True:
            if argument.startswith("--allow-write"):
                allow_write = True
                argument = argument[len("--allow-write"):].strip()
            elif argument.startswith("--allow-web"):
                allow_web = True
                argument = argument[len("--allow-web"):].strip()
            elif argument.startswith("--verbose"):
                verbose = True
                argument = argument[len("--verbose"):].strip()
            else:
                break
        if not argument:
            raise ValueError(AGENT_USAGE)
        if self._pending is not None:
            self._pending = None
        if self._pending_agent is not None:
            self._pending_agent = None
        return self._agent_run(argument, allow_write, allow_web=allow_web, verbose=verbose)

    # -- Day 3: persistent tasks -------------------------------------------------

    @staticmethod
    def _step_to_dict(step: AgentStep) -> dict[str, Any]:
        return {
            "index": step.index, "tool": step.tool, "args": step.args,
            "status": step.status, "output": step.output, "error": step.error,
            "attempts": step.attempts, "verified": step.verified, "note": step.note,
        }

    @staticmethod
    def _dict_to_step(data: dict[str, Any]) -> AgentStep:
        args = data.get("args", {})
        return AgentStep(
            index=int(data.get("index", 0)),
            tool=str(data.get("tool", "")),
            args=dict(args) if isinstance(args, dict) else {},
            status=str(data.get("status", "pending")),
            output=str(data.get("output", "")),
            error=str(data.get("error", "")),
            attempts=int(data.get("attempts", 0)),
            verified=bool(data.get("verified", False)),
            note=str(data.get("note", "")),
        )

    def _task_command(self, message: str) -> str | None:
        command, _, argument = message.partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command == "/tasks":
            limit = 10
            if argument:
                try:
                    limit = int(argument)
                except ValueError as exc:
                    raise ValueError("Usage: /tasks [limit]") from exc
            rows = self.tasks.list_recent(limit)
            if not rows:
                return "No agent tasks recorded yet."
            lines = [
                f"#{row['id']} [{row['status']}] {(row['goal'] or '')[:80]} ({row['updated_at']})"
                for row in rows
            ]
            return "Recent agent tasks:\n" + "\n".join(lines)
        if command == "/resume":
            try:
                resume_id = int(argument)
            except ValueError as exc:
                raise ValueError("Usage: /resume <task ID>") from exc
            saved = self.tasks.load_resumable(resume_id)
            if saved is None:
                return f"Task #{resume_id} is not resumable (missing, finished, or cancelled)."
            steps = [self._dict_to_step(d) for d in saved.get("steps", []) if isinstance(d, dict)]
            pending = saved.get("pending", {})
            if not isinstance(pending, dict) or not pending.get("tool"):
                return f"Task #{resume_id} has no pending approval to resume."
            flags = saved.get("flags", {})
            self._pending_agent = {
                "goal": str(saved.get("goal", "")),
                "allow_write": bool(flags.get("allow_write", False)),
                "allow_web": bool(flags.get("allow_web", False)),
                "verbose": bool(flags.get("verbose", False)),
                "steps": steps,
                "step": {"tool": str(pending["tool"]), "args": dict(pending.get("args", {}))},
                "task_id": resume_id,
            }
            return (
                f"Resumed task #{resume_id} ('{str(saved.get('goal', ''))[:80]}'). "
                f"Step {len(steps) + 1} wants to run '{pending['tool']}', which needs approval. "
                "Reply yes to allow this step and continue, or no to drop the run."
            )
        return None

    def _skill_command(self, message: str) -> str | None:
        command, _, argument = message.partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command == "/skills":
            skills = self.skills.list_skills()
            if not skills:
                return "No saved skills yet. Use /skill save <name> after a successful task."
            lines = [
                f"{s['name']} v{s['version']} [{s['status']}]"
                + (f" (from task #{s['source_task_id']})" if s["source_task_id"] else "")
                for s in skills
            ]
            return "Saved skills:\n" + "\n".join(lines)
        if command == "/skill":
            sub, _, rest = argument.partition(" ")
            sub = sub.casefold()
            rest = rest.strip()
            if sub == "save":
                return self._skill_save(rest)
            if sub == "test":
                if not rest:
                    raise ValueError("Usage: /skill test <name>")
                skill = self.skills.get_latest(rest)
                if skill is None:
                    return f"No skill named '{rest}'."
                error = SkillStore.validate_workflow(skill["workflow"], self.tools)
                if error is not None:
                    return f"Skill '{rest}' v{skill['version']} failed validation: {error}"
                self.skills.mark_tested(rest, int(skill["version"]))
                return f"Skill '{rest}' v{skill['version']} passed validation and is marked tested."
            if sub == "enable":
                if not rest:
                    raise ValueError("Usage: /skill enable <name>")
                try:
                    self.skills.enable(rest)
                except (KeyError, ValueError) as exc:
                    return str(exc)
                skill = self.skills.get_latest(rest)
                return f"Skill '{rest}' v{skill['version']} is now enabled."
            raise ValueError("Usage: /skill save <name> | /skill test <name> | /skill enable <name>")
        return None

    def _skill_save(self, name: str) -> str:
        if not name:
            raise ValueError("Usage: /skill save <name>")
        steps = [s for s in self._last_steps if s.status == "ok"]
        if not steps:
            return "No successful steps in the last task to save as a skill."
        workflow = [{"tool": s.tool, "args": s.args} for s in steps]
        error = SkillStore.validate_workflow(workflow, self.tools)
        if error is not None:
            return f"Last task cannot become a skill: {error}"
        _, version = self.skills.save_draft(
            name, workflow, source_task_id=self._last_task_id,
            note=f"Captured from task #{self._last_task_id}: {self._last_goal[:120]}",
        )
        return (
            f"Saved skill '{name}' v{version} as a draft ({len(workflow)} step(s), from task "
            f"#{self._last_task_id}). Test it with /skill test {name} before enabling."
        )

    def _agent_run(
        self,
        goal: str,
        allow_write: bool,
        resume_steps: list[AgentStep] | None = None,
        resume_confirmed: dict[str, Any] | None = None,
        first_reply: str | None = None,
        allow_web: bool = False,
        verbose: bool = False,
        task_id: int | None = None,
    ) -> str:
        started = time.monotonic()
        steps: list[AgentStep] = list(resume_steps or [])
        flags = {"allow_write": allow_write, "allow_web": allow_web, "verbose": verbose}
        if task_id is None:
            task_id = self.tasks.create(goal, flags)
        self._last_task_id = task_id
        self._last_goal = goal
        relevant = self.long_term.retrieve_relevant(goal)
        memories = [f"[{entry['category']}] {entry['content']}" for entry in relevant]
        system = build_agent_system(goal, [spec.usage() for spec in self.tools.specs()], memories)
        history: list[dict[str, str]] = [{"role": "system", "content": system}]
        for row in self.memory.get_messages(limit=12)[:-1]:
            history.append({"role": row["role"], "content": row["content"]})
        for step in steps:
            history.append({"role": "assistant", "content": f"Tool request: {step.tool} {json.dumps(step.args)}"})
            history.append({"role": "user", "content": self._step_feedback(step)})
        self._report_progress(f"Agent started: {goal[:80]}")

        def elapsed() -> int:
            return int((time.monotonic() - started) * 1000)

        def aged_out() -> bool:
            return (time.monotonic() - started) > self._agent_timeout

        # Resume path: the paused write was already approved via /confirm.
        if resume_confirmed is not None:
            step = AgentStep(
                index=len(steps) + 1,
                tool=str(resume_confirmed["name"]),
                args=dict(resume_confirmed["args"]),
            )
            self._execute_agent_step(step, confirmed=True)
            steps.append(step)
            history.append({"role": "assistant", "content": f"Tool request: {step.tool} {json.dumps(step.args)}"})
            history.append({"role": "user", "content": self._step_feedback(step)})

        reason = "finished"
        final = ""
        pending_first = first_reply
        turns = 0
        while turns < self._agent_max_iterations:
            if self._agent_cancel.is_set():
                reason = "cancelled"
                final = "The run was cancelled. Completed steps are listed above."
                break
            if aged_out():
                reason = "task timeout"
                final = "The task budget ran out. Completed steps are listed above."
                break
            if pending_first is not None:
                reply = pending_first
                pending_first = None
            else:
                prompt = goal if not steps else "Continue the task."
                try:
                    reply = self.model.respond(prompt, history)
                except Exception as exc:
                    if not steps and turns == 0 and resume_confirmed is None:
                        raise
                    reason = "model error"
                    final = f"The model could not continue: {exc}. Completed steps are listed above."
                    break
            turns += 1
            if not reply.strip():
                reason = "model error"
                final = "The model returned an empty response. Completed steps are listed above."
                break
            try:
                request = parse_tool_request(reply)
            except ValueError as exc:
                step = AgentStep(index=len(steps) + 1, tool="(unreadable request)", status="failed", error=str(exc))
                steps.append(step)
                history.append({"role": "assistant", "content": reply.strip()[:500]})
                history.append({"role": "user", "content": self._step_feedback(step)})
                continue
            if request is None:
                reason = "finished"
                final = reply.strip()
                break
            name, args = request
            step = AgentStep(index=len(steps) + 1, tool=name, args=args)
            self._report_progress(f"Agent step {step.index}: running {name}…")
            probe = self.tools.execute(name, args)
            if probe.confirmation_required and not self._step_authorized(name, allow_write, allow_web):
                self._pending_agent = {
                    "goal": goal,
                    "allow_write": allow_write,
                    "allow_web": allow_web,
                    "verbose": verbose,
                    "steps": steps,
                    "step": {"tool": name, "args": args},
                    "task_id": task_id,
                }
                self.tasks.save_pause(
                    task_id,
                    [self._step_to_dict(s) for s in steps],
                    {"tool": name, "args": args},
                    flags,
                )
                self._last_steps = list(steps)
                partial = summarize(steps, "Waiting for confirmation.", "paused for confirmation", elapsed())
                return (
                    partial
                    + f"\n\nStep {step.index} wants to run '{name}', which needs approval. "
                    f"Reply yes to allow this step and continue, or no to drop the run. (task #{task_id})"
                )
            if probe.confirmation_required or (not probe.ok and "timed out" in probe.error):
                # Pre-approved just now, or a transient timeout worth one retry.
                self._execute_agent_step(step, confirmed=True)
            else:
                # Read-only tools already ran once in the probe; reuse, don't repeat.
                self._record_result(step, probe)
            steps.append(step)
            history.append({"role": "assistant", "content": f"Tool request: {step.tool} {json.dumps(step.args)}"})
            history.append({"role": "user", "content": self._step_feedback(step)})
        if reason == "finished" and not final:
            reason = "max iterations"
            final = (
                f"Reached the limit of {self._agent_max_iterations} model turn(s). "
                "Completed steps are listed above; refine the goal and try again."
            )

        self._agent_cancel.clear()
        self._report_progress("Agent ready.")
        self._last_steps = list(steps)
        status_map = {
            "finished": "finished" if not any(s.status == "failed" for s in steps) else "finished",
            "cancelled": "cancelled",
            "task timeout": "timeout",
            "model error": "model-error",
            "max iterations": "max-iterations",
        }
        try:
            self.tasks.finish(
                task_id,
                status_map.get(reason, "finished"),
                [self._step_to_dict(s) for s in steps],
                final,
            )
        except Exception:
            logger.warning("Could not persist task run", exc_info=True)
        return summarize(steps, final, reason, elapsed(), verbose)

    @staticmethod
    def _step_authorized(tool: str, allow_write: bool, allow_web: bool) -> bool:
        """Pre-approval covers only the flags the run was started with; default deny.

        repo.test always needs explicit approval: it executes repository code.
        """
        if tool.startswith("web."):
            return allow_web
        if tool in {"workspace.write", "repo.edit"}:
            return allow_write
        return False

    def _record_result(self, step: AgentStep, result: Any) -> None:
        """Store a tool outcome on a step, verifying writes by re-reading."""
        if not result.ok:
            step.status = "failed"
            step.error = result.error
            step.output = ""
            return
        step.status = "ok"
        step.output = result.output
        step.error = ""
        if step.tool == "workspace.write":
            content = step.args.get("content")
            path = step.args.get("path")
            if isinstance(content, str) and isinstance(path, str):
                verified, note = verify_write(self.tools, path, content)
                step.verified = verified
                step.note = note
            else:
                step.note = "write arguments missing; could not verify"
        else:
            step.verified = True
            step.note = "result inspected"

    def _execute_agent_step(self, step: AgentStep, confirmed: bool) -> None:
        """Run one tool step with bounded retries on timeouts."""
        attempts = 1 + self._agent_retry
        for attempt in range(1, attempts + 1):
            step.attempts = attempt
            result = self.tools.execute(step.tool, step.args, confirmed=confirmed)
            if result.confirmation_required:
                step.status = "failed"
                step.error = "Permission denied: confirmation is still required."
                return
            if result.ok:
                self._record_result(step, result)
                return
            # Retry only transient timeouts, never validation or permission errors.
            if "timed out" in result.error and attempt < attempts:
                step.error = result.error
                continue
            self._record_result(step, result)
            return

    @staticmethod
    def _step_feedback(step: AgentStep) -> str:
        if step.status == "ok":
            extra = f" Verification: {step.note}." if step.note else ""
            return f"Tool result (ok): {truncate_feed(step.output)}.{extra}"
        return f"Tool result (failed): {truncate_feed(step.error or 'unknown error')}. Adjust or finish with a summary."
