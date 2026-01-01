"""Solution execution utilities for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import asyncio
import base64
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

# Third-party (alphabetical)
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.messages import BuiltinToolReturnPart, ModelResponse

# Local imports (core first, then alphabetical)
from agent_k.infra.providers import get_model

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ('BASELINE_SCORE_PATTERN', 'ExecutionResult', 'execute_solution', 'parse_baseline_score')

BASELINE_SCORE_PATTERN: Final[re.Pattern[str]] = re.compile(r'Baseline .*? score:\s*(-?[0-9.]+)', re.IGNORECASE)
_CODE_EXECUTION_SYSTEM_PROMPT: Final[str] = (
    'You are a code execution runner. Always call the code_execution tool with the exact '
    'Python code provided by the user message, without modification. After the tool '
    "returns, respond with the single word 'done'."
)
_DEFAULT_MAX_INLINE_DATA_BYTES: Final[int] = 100_000
_EXECUTION_DATA_FILES: Final[tuple[str, ...]] = ('train.csv', 'test.csv', 'sample_submission.csv')
_SENSITIVE_ENV_TOKENS: Final[tuple[str, ...]] = (
    'KEY',
    'TOKEN',
    'SECRET',
    'PASSWORD',
    'PASS',
    'CREDENTIAL',
    'OPENAI',
    'ANTHROPIC',
    'OPENROUTER',
    'KAGGLE',
    'LOGFIRE',
)

_CODE_EXECUTION_AGENT_CACHE: dict[str, Agent[None, str]] = {}


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of executing a solution script."""

    returncode: int
    stdout: str
    stderr: str
    runtime_ms: int
    timed_out: bool


async def execute_solution(
    code: str,
    work_path: Path,
    *,
    timeout_seconds: float | None = None,
    env: dict[str, str] | None = None,
    use_builtin_code_execution: bool = False,
    model_spec: str | None = None,
    max_inline_data_bytes: int = _DEFAULT_MAX_INLINE_DATA_BYTES,
) -> ExecutionResult:
    """Execute solution code in a working directory."""
    if use_builtin_code_execution:
        tool_result = await _execute_with_builtin_tool(
            code, work_path, env=env, model_spec=model_spec, max_inline_data_bytes=max_inline_data_bytes
        )
        if tool_result is not None:
            return tool_result

    return await _execute_solution_local(code, work_path, timeout_seconds=timeout_seconds, env=env)


def parse_baseline_score(output: str) -> float | None:
    """Parse baseline score from solution output."""
    match = BASELINE_SCORE_PATTERN.search(output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


async def _execute_solution_local(
    code: str, work_path: Path, *, timeout_seconds: float | None, env: dict[str, str] | None
) -> ExecutionResult:
    solution_path = work_path / 'solution.py'
    solution_path.write_text(code, encoding='utf-8')

    exec_env = _sanitize_env(env, work_path=work_path)

    start_time = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-I',
        str(solution_path),
        cwd=str(work_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=exec_env,
    )

    timed_out = False
    try:
        if timeout_seconds is None:
            stdout, stderr = await process.communicate()
        else:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout, stderr = await process.communicate()

    runtime_ms = int((time.perf_counter() - start_time) * 1000)
    return ExecutionResult(
        returncode=process.returncode if process.returncode is not None else 1,
        stdout=stdout.decode('utf-8', errors='ignore'),
        stderr=stderr.decode('utf-8', errors='ignore'),
        runtime_ms=runtime_ms,
        timed_out=timed_out,
    )


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
        'ModelSettings', {'temperature': 0.0, 'max_tokens': 256, 'openai_include_code_execution_outputs': True}
    )
    agent = Agent(
        model=get_model(model_spec),
        output_type=str,
        instructions=_CODE_EXECUTION_SYSTEM_PROMPT,
        builtin_tools=[CodeExecutionTool()],
        model_settings=model_settings,
        retries=1,
        output_retries=0,
        name='code_executor',
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
        payloads[filename] = base64.b64encode(data).decode('ascii')
    return payloads


def _build_code_execution_script(code: str, *, env: dict[str, str] | None, inline_files: dict[str, str]) -> str:
    lines: list[str] = ['import base64', 'from pathlib import Path']

    if env:
        lines.append('import os')
        lines.extend(f'os.environ[{key!r}] = {value!r}' for key, value in env.items())
    if inline_files:
        lines.append('FILES = {')
        for name in sorted(inline_files):
            payload = inline_files[name]
            lines.append(f'    {name!r}: {payload!r},')
        lines.extend(
            ('}', 'for name, payload in FILES.items():', '    Path(name).write_bytes(base64.b64decode(payload))')
        )
    lines.append(code)
    return '\n'.join(lines)


def _extract_code_execution_result(messages: list[Any]) -> dict[str, Any] | None:
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, BuiltinToolReturnPart) and part.tool_name == CodeExecutionTool.kind:
                if isinstance(part.content, dict):
                    return part.content
                return {'return_value': part.content}
    return None


def _parse_code_execution_result(content: dict[str, Any], runtime_ms: int) -> ExecutionResult:
    error_code = content.get('error_code')
    if error_code:
        return ExecutionResult(
            returncode=1,
            stdout='',
            stderr=f'Code execution error: {error_code}',
            runtime_ms=runtime_ms,
            timed_out=error_code == 'execution_time_exceeded',
        )

    stdout = content.get('stdout') or ''
    stderr = content.get('stderr') or ''
    returncode = content.get('return_code')
    if returncode is None:
        returncode = content.get('returncode', content.get('exit_code', 0))
    try:
        returncode_value = int(returncode)
    except (TypeError, ValueError):
        returncode_value = 1
    return ExecutionResult(
        returncode=returncode_value, stdout=str(stdout), stderr=str(stderr), runtime_ms=runtime_ms, timed_out=False
    )


def _sanitize_env(extra_env: dict[str, str] | None, *, work_path: Path) -> dict[str, str]:
    sanitized = {key: value for key, value in os.environ.items() if not _is_sensitive_env_key(key)}
    if extra_env:
        sanitized.update(extra_env)
    sanitized.setdefault('PYTHONNOUSERSITE', '1')
    sanitized.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    # Always override HOME to isolate execution in the work directory
    sanitized['HOME'] = str(work_path)
    return sanitized


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(token in normalized for token in _SENSITIVE_ENV_TOKENS)


def _supports_code_execution(model_spec: str) -> bool:
    return model_spec.startswith(('anthropic:', 'openai:', 'google:'))
