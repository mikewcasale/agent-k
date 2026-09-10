"""Solution execution utilities for AGENT-K.

@notice: |
    Solution execution utilities for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.solution
    provides:
        - agent_k.core.solution
    pattern: solution-models

@agent-guidance:
    do:
        - "Use agent_k.core.solution as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import contextlib
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import logfire
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.messages import BuiltinToolReturnPart, ModelResponse

from agent_k.core.sage import Doc
from agent_k.infra.providers import get_model

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "BASELINE_SCORE_PATTERN",
    "MAX_CAPTURED_OUTPUT_BYTES",
    "ExecutionResult",
    "execute_solution",
    "parse_baseline_score",
)

BASELINE_SCORE_PATTERN: Final[re.Pattern[str]] = re.compile(r"Baseline .*? score:\s*(-?[0-9.]+)", re.IGNORECASE)
# Per-stream capture ceiling. Consumers keep at most a few kilobytes of stdout/stderr, so this
# leaves ample room for fold scores and tracebacks while bounding parent memory for runaway
# solutions.
MAX_CAPTURED_OUTPUT_BYTES: Final[int] = 262_144
# Share of the budget spent on the head. Warnings are emitted early, while fold scores, the
# baseline score line, and tracebacks land at the end, so the tail keeps the larger share.
_CAPTURE_HEAD_FRACTION: Final[float] = 0.25
_CAPTURE_READ_CHUNK_BYTES: Final[int] = 65_536
_CAPTURE_DRAIN_GRACE_SECONDS: Final[float] = 5.0
_PROCESS_REAP_GRACE_SECONDS: Final[float] = 5.0
_CODE_EXECUTION_SYSTEM_PROMPT: Final[str] = (
    "You are a code execution runner. Always call the code_execution tool with the exact "
    "Python code provided by the user message, without modification. After the tool "
    "returns, respond with the single word 'done'."
)
_DEFAULT_MAX_INLINE_DATA_BYTES: Final[int] = 100_000
_EXECUTION_DATA_FILES: Final[tuple[str, ...]] = ("train.csv", "test.csv", "sample_submission.csv")
_KAGGLE_INPUT_PREFIX: Final[str] = "/kaggle/input"
_SENSITIVE_ENV_TOKENS: Final[tuple[str, ...]] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASS",
    "CREDENTIAL",
    "OPENAI",
    "ANTHROPIC",
    "OPENROUTER",
    "KAGGLE",
    "LOGFIRE",
)

_CODE_EXECUTION_AGENT_CACHE: dict[str, Agent[None, str]] = {}


@dataclass(slots=True)
class _BoundedCapture:
    """Fixed-memory capture of a byte stream that keeps the head and the tail.

    @notice: |
        Accumulates at most ``limit`` bytes from a stream while counting everything it sees.

    @dev: |
        The head preserves early output such as import warnings; the tail preserves fold scores,
        the baseline score line, and tracebacks. Middle output is dropped and replaced by a notice.
        Trimming is amortised by letting the tail grow to twice its share before slicing.

        @pattern:
            name: ring-buffer
            rationale: "Caps parent memory for solutions that write unbounded output."
            violations: "Reading a whole pipe into memory lets one candidate exhaust the host."
    """

    limit: int
    total_bytes: int = 0
    _head: bytearray = field(default_factory=bytearray, repr=False)
    _tail: bytearray = field(default_factory=bytearray, repr=False)

    @property
    def _head_limit(self) -> int:
        return int(self.limit * _CAPTURE_HEAD_FRACTION)

    @property
    def _tail_limit(self) -> int:
        return self.limit - self._head_limit

    @property
    def dropped_bytes(self) -> int:
        """Bytes the stream produced that the excerpt will not contain."""
        retained_tail = min(len(self._tail), self._tail_limit)
        return max(0, self.total_bytes - len(self._head) - retained_tail)

    @property
    def truncated(self) -> bool:
        """Whether the stream produced more bytes than the capture budget."""
        return self.dropped_bytes > 0

    def feed(self, chunk: bytes) -> None:
        """Record a chunk, keeping memory bounded by the configured limit.

        @dev: |
            Fills the head first, then appends to the tail and trims it from the left.
        """
        self.total_bytes += len(chunk)
        head_room = self._head_limit - len(self._head)
        if head_room > 0:
            self._head.extend(chunk[:head_room])
            chunk = chunk[head_room:]
            if not chunk:
                return

        tail_limit = self._tail_limit
        self._tail.extend(chunk)
        if len(self._tail) > tail_limit * 2:
            del self._tail[: len(self._tail) - tail_limit]

    def render(self) -> str:
        """Decode the captured excerpt, marking any dropped middle section.

        @dev: |
            Decoding is lossy on purpose: solutions may emit arbitrary bytes.
        """
        dropped = self.dropped_bytes
        tail_limit = self._tail_limit
        if len(self._tail) > tail_limit:
            del self._tail[: len(self._tail) - tail_limit]

        head_text = self._head.decode("utf-8", errors="ignore")
        tail_text = self._tail.decode("utf-8", errors="ignore")
        if dropped <= 0:
            return head_text + tail_text
        return f"{head_text}\n... [{dropped} bytes of output dropped] ...\n{tail_text}"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of executing a solution script.

    @notice: |
        Result of executing a solution script.

    @dev: |
        ``stdout``/``stderr`` hold the captured excerpt, which is bounded by the capture budget.
        ``stdout_bytes``/``stderr_bytes`` report how much the process actually wrote, and
        ``output_truncated`` is set when either stream exceeded the budget.

        @pattern:
            name: execution-result
            rationale: "Standardizes subprocess outputs for solution runs."
            violations: "Tuple returns drop context for debugging."
    """

    returncode: int
    stdout: str
    stderr: str
    runtime_ms: int
    timed_out: bool
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    output_truncated: bool = False


async def execute_solution(
    code: str,
    work_path: Path,
    *,
    timeout_seconds: float | None = None,
    env: dict[str, str] | None = None,
    use_builtin_code_execution: bool = False,
    model_spec: str | None = None,
    max_inline_data_bytes: int = _DEFAULT_MAX_INLINE_DATA_BYTES,
    max_output_bytes: Annotated[int, Doc("Per-stream capture ceiling in bytes.")] = MAX_CAPTURED_OUTPUT_BYTES,
) -> ExecutionResult:
    """Execute solution code in a working directory.

    @notice: |
        Runs Python code in an isolated subprocess with timeout support.

    @dev: |
        Normalizes Kaggle paths, sanitizes environment, and captures output.
        Supports builtin code execution tool or local subprocess execution.
        Output capture is bounded by ``max_output_bytes`` per stream.
    """
    normalized_code = _normalize_kaggle_paths(code)
    if use_builtin_code_execution:
        tool_result = await _execute_with_builtin_tool(
            normalized_code, work_path, env=env, model_spec=model_spec, max_inline_data_bytes=max_inline_data_bytes
        )
        if tool_result is not None:
            return tool_result

    return await _execute_solution_local(
        normalized_code, work_path, timeout_seconds=timeout_seconds, env=env, max_output_bytes=max_output_bytes
    )


def parse_baseline_score(output: str) -> float | None:
    """Parse baseline score from solution output.

    @notice: |
        Extracts numeric score from "Baseline ... score: X.XX" pattern.

    @dev: |
        Returns None if pattern not found or value cannot be parsed.
    """
    if match := BASELINE_SCORE_PATTERN.search(output):
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _normalize_kaggle_paths(code: str) -> str:
    if _KAGGLE_INPUT_PREFIX not in code:
        return code
    return code.replace(_KAGGLE_INPUT_PREFIX, ".")


async def _execute_solution_local(
    code: str, work_path: Path, *, timeout_seconds: float | None, env: dict[str, str] | None, max_output_bytes: int
) -> ExecutionResult:
    solution_path = work_path / "solution.py"
    solution_path.write_text(code, encoding="utf-8")

    exec_env = _sanitize_env(env, work_path=work_path)

    start_time = time.perf_counter()
    process_kwargs: dict[str, Any] = {
        "cwd": str(work_path),
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": exec_env,
    }
    if os.name == "posix":
        process_kwargs["start_new_session"] = True

    process = await asyncio.create_subprocess_exec(sys.executable, "-I", str(solution_path), **process_kwargs)

    stdout_capture = _BoundedCapture(limit=max_output_bytes)
    stderr_capture = _BoundedCapture(limit=max_output_bytes)
    timed_out = await _await_process(process, stdout_capture, stderr_capture, timeout_seconds=timeout_seconds)

    runtime_ms = int((time.perf_counter() - start_time) * 1000)
    truncated = stdout_capture.truncated or stderr_capture.truncated
    if truncated:
        logfire.warning(
            "solution_output_truncated",
            stdout_bytes=stdout_capture.total_bytes,
            stderr_bytes=stderr_capture.total_bytes,
            limit_bytes=max_output_bytes,
        )

    return ExecutionResult(
        returncode=process.returncode if process.returncode is not None else 1,
        stdout=stdout_capture.render(),
        stderr=stderr_capture.render(),
        runtime_ms=runtime_ms,
        timed_out=timed_out,
        stdout_bytes=stdout_capture.total_bytes,
        stderr_bytes=stderr_capture.total_bytes,
        output_truncated=truncated,
    )


async def _await_process(
    process: asyncio.subprocess.Process,
    stdout_capture: _BoundedCapture,
    stderr_capture: _BoundedCapture,
    *,
    timeout_seconds: float | None,
) -> bool:
    """Wait for a solution subprocess while draining both pipes into bounded captures.

    @dev: |
        Draining runs concurrently with the wait so a solution that writes more than the OS pipe
        buffer cannot deadlock. Returns whether the timeout fired.
    """
    drains = [
        asyncio.create_task(_drain_stream(process.stdout, stdout_capture)),
        asyncio.create_task(_drain_stream(process.stderr, stderr_capture)),
    ]

    timed_out = False
    try:
        if timeout_seconds is None:
            await process.wait()
        else:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        logfire.warning("solution_execution_timeout", timeout_seconds=timeout_seconds, pid=process.pid)
        _terminate_process(process)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_PROCESS_REAP_GRACE_SECONDS)

    await _finish_drains(drains)
    return timed_out


async def _drain_stream(stream: asyncio.StreamReader | None, capture: _BoundedCapture) -> None:
    if stream is None:
        return
    while chunk := await stream.read(_CAPTURE_READ_CHUNK_BYTES):
        capture.feed(chunk)


async def _finish_drains(drains: list[asyncio.Task[None]]) -> None:
    """Let the pipe readers reach EOF, then cancel any that outlive the grace period.

    @dev: |
        Captures are mutated in place, so cancelling a straggling reader still preserves whatever
        it managed to collect. A reader can outlive the process when a grandchild inherited the
        pipe and escaped termination.
    """
    done, still_running = await asyncio.wait(drains, timeout=_CAPTURE_DRAIN_GRACE_SECONDS)
    for drain in still_running:
        drain.cancel()
    if still_running:
        await asyncio.wait(still_running)
        logfire.warning("solution_output_drain_abandoned", readers=len(still_running))
    for drain in done:
        if drain.cancelled():
            continue
        if (error := drain.exception()) is not None:
            logfire.warning("solution_output_drain_failed", error=str(error))


async def _execute_with_builtin_tool(
    code: str, work_path: Path, *, env: dict[str, str] | None, model_spec: str | None, max_inline_data_bytes: int
) -> ExecutionResult | None:
    if model_spec is None:
        return None
    if not _supports_code_execution(model_spec):
        return None

    inline_files = _load_inline_files(work_path, max_inline_data_bytes=max_inline_data_bytes)
    if inline_files is None:
        return None

    script = _build_code_execution_script(code, env=env, inline_files=inline_files)
    agent = _get_code_execution_agent(model_spec)

    start_time = time.perf_counter()
    try:
        run_result = await agent.run(script)
    except Exception:
        return None

    runtime_ms = int((time.perf_counter() - start_time) * 1000)
    tool_content = _extract_code_execution_result(run_result.all_messages())
    if tool_content is None:
        return None
    return _parse_code_execution_result(tool_content, runtime_ms)


def _get_code_execution_agent(model_spec: str) -> Agent[None, str]:
    cached = _CODE_EXECUTION_AGENT_CACHE.get(model_spec)
    if cached is not None:
        return cached

    model_settings = cast(
        "ModelSettings", {"temperature": 0.0, "max_tokens": 256, "openai_include_code_execution_outputs": True}
    )
    agent = Agent(
        model=get_model(model_spec),
        output_type=str,
        instructions=_CODE_EXECUTION_SYSTEM_PROMPT,
        builtin_tools=[CodeExecutionTool()],
        model_settings=model_settings,
        retries=1,
        output_retries=0,
        name="code_executor",
        instrument=True,
    )
    _CODE_EXECUTION_AGENT_CACHE[model_spec] = agent
    return agent


def _load_inline_files(work_path: Path, *, max_inline_data_bytes: int) -> dict[str, str] | None:
    total_bytes = 0
    payloads: dict[str, str] = {}
    for filename in _EXECUTION_DATA_FILES:
        file_path = work_path / filename
        if not file_path.exists():
            return None
        file_size = file_path.stat().st_size
        total_bytes += file_size
        if total_bytes > max_inline_data_bytes:
            return None
        data = file_path.read_bytes()
        payloads[filename] = base64.b64encode(data).decode("ascii")
    return payloads


def _build_code_execution_script(code: str, *, env: dict[str, str] | None, inline_files: dict[str, str]) -> str:
    lines: list[str] = ["import base64", "from pathlib import Path"]

    if env:
        lines.append("import os")
        lines.extend(f"os.environ[{key!r}] = {value!r}" for key, value in env.items())
    if inline_files:
        lines.append("FILES = {")
        for name in sorted(inline_files):
            payload = inline_files[name]
            lines.append(f"    {name!r}: {payload!r},")
        lines.extend(
            ("}", "for name, payload in FILES.items():", "    Path(name).write_bytes(base64.b64decode(payload))")
        )
    lines.append(code)
    return "\n".join(lines)


def _extract_code_execution_result(messages: list[Any]) -> dict[str, Any] | None:
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, BuiltinToolReturnPart) and part.tool_name == CodeExecutionTool.kind:
                if isinstance(part.content, dict):
                    return part.content
                return {"return_value": part.content}
    return None


def _parse_code_execution_result(content: dict[str, Any], runtime_ms: int) -> ExecutionResult:
    if error_code := content.get("error_code"):
        return _bounded_result(
            returncode=1,
            stdout="",
            stderr=f"Code execution error: {error_code}",
            runtime_ms=runtime_ms,
            timed_out=error_code == "execution_time_exceeded",
        )

    stdout = content.get("stdout") or ""
    stderr = content.get("stderr") or ""
    returncode = content.get("return_code")
    if returncode is None:
        returncode = content.get("returncode", content.get("exit_code", 0))
    try:
        returncode_value = int(returncode)
    except (TypeError, ValueError):
        returncode_value = 1
    return _bounded_result(
        returncode=returncode_value, stdout=str(stdout), stderr=str(stderr), runtime_ms=runtime_ms, timed_out=False
    )


def _bounded_result(*, returncode: int, stdout: str, stderr: str, runtime_ms: int, timed_out: bool) -> ExecutionResult:
    """Build an execution result from strings under the same capture budget as the local runner.

    @dev: |
        Keeps remote code-execution output bounded so both execution paths report byte counts and
        truncation the same way.
    """
    captures: list[_BoundedCapture] = []
    for text in (stdout, stderr):
        capture = _BoundedCapture(limit=MAX_CAPTURED_OUTPUT_BYTES)
        capture.feed(text.encode("utf-8", errors="ignore"))
        captures.append(capture)

    stdout_capture, stderr_capture = captures
    return ExecutionResult(
        returncode=returncode,
        stdout=stdout_capture.render(),
        stderr=stderr_capture.render(),
        runtime_ms=runtime_ms,
        timed_out=timed_out,
        stdout_bytes=stdout_capture.total_bytes,
        stderr_bytes=stderr_capture.total_bytes,
        output_truncated=stdout_capture.truncated or stderr_capture.truncated,
    )


def _sanitize_env(extra_env: dict[str, str] | None, *, work_path: Path) -> dict[str, str]:
    sanitized = {key: value for key, value in os.environ.items() if not _is_sensitive_env_key(key)}
    if extra_env:
        sanitized.update(extra_env)
    sanitized.setdefault("PYTHONNOUSERSITE", "1")
    sanitized.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Always override HOME to isolate execution in the work directory
    sanitized["HOME"] = str(work_path)
    return sanitized


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(token in normalized for token in _SENSITIVE_ENV_TOKENS)


def _supports_code_execution(model_spec: str) -> bool:
    return model_spec.startswith("openai:")


def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    process.kill()
