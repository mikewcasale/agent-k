#!/usr/bin/env python3
"""AGENT-K Multi-Agent Playbook Demo.

This demonstrates the full AGENT-K multi-agent flow using built-in tools:

1. WebSearchTool - Built-in web search
2. MCPServerTool(kaggle) - Kaggle API via KaggleToolset
3. MemoryTool - Built-in memory (Anthropic only) with file-backed backend

The demo runs through:
- LOBBYIST: Competition discovery using web search + Kaggle API
- SCIENTIST: Research phase using web search + memory
- Optional memory persistence across agents (Anthropic only)

Usage:
    # Claude Haiku (Anthropic)
    python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

    # GPT-4o (OpenAI Responses)
    python examples/multi_agent_playbook.py --model openai:gpt-4o

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel

__all__ = ()

console = Console()

# Supported models for the demo (built-in tools required)
SUPPORTED_MODELS = [
    "anthropic:claude-3-haiku-20240307",
    "openai:gpt-4o",
]


# =============================================================================
# Agent Definitions - Use str output for broad model compatibility
# =============================================================================


def _bootstrap() -> None:
    """Configure sys.path and load environment variables."""
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root))

    env_path = backend_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def create_lobbyist_agent(
    model_spec: str,
    kaggle_toolset: Any,
    builtin_tools: list[Any],
    memory_enabled: bool,
) -> Agent[Any, str]:
    """Create the LOBBYIST agent for competition discovery."""

    from agent_k.core.deps import KaggleDeps
    from agent_k.infra.providers import get_model

    model = get_model(model_spec)

    memory_instructions = ""
    if memory_enabled:
        memory_instructions = (
            "\n- memory: store shared notes (create/view commands)\n"
            "\nWORKFLOW:\n"
            "1. FIRST call kaggle_search_competitions() to find active competitions\n"
            "2. THEN call web_search() to find recent Kaggle news\n"
            "3. THEN use memory to create shared/target_competition.md with the best competition\n"
        )
    else:
        memory_instructions = (
            "\nWORKFLOW:\n"
            "1. FIRST call kaggle_search_competitions() to find active competitions\n"
            "2. THEN call web_search() to find recent Kaggle news\n"
            "3. THEN summarize the best competition clearly for the SCIENTIST\n"
        )

    instructions = (
        "You are the LOBBYIST agent in the AGENT-K multi-agent system.\n\n"
        "Your mission is to discover Kaggle competitions that match the user's criteria.\n\n"
        "AVAILABLE TOOLS:\n"
        "- kaggle_search_competitions: Search Kaggle API for competitions\n"
        "- kaggle_get_competition: Get details about a specific competition\n"
        "- web_search: Built-in web search\n"
        f"{memory_instructions}"
        "\nAt the end, provide a summary of your findings.\n"
    )

    return Agent(
        model,
        deps_type=KaggleDeps,
        output_type=str,  # Simple string output for broad compatibility
        toolsets=[kaggle_toolset],
        builtin_tools=builtin_tools,
        instructions=instructions,
        name="lobbyist",
        retries=5,
    )


def create_scientist_agent(
    model_spec: str,
    kaggle_toolset: Any,
    builtin_tools: list[Any],
    memory_enabled: bool,
) -> Agent[Any, str]:
    """Create the SCIENTIST agent for research."""

    from agent_k.core.deps import KaggleDeps
    from agent_k.infra.providers import get_model

    model = get_model(model_spec)

    memory_instructions = ""
    if memory_enabled:
        memory_instructions = (
            "\n- memory: read shared notes (view commands)\n"
            "\nWORKFLOW:\n"
            "1. FIRST use memory to view shared/target_competition.md\n"
            "2. THEN call kaggle_get_leaderboard() to analyze current standings\n"
            "3. THEN call web_search() with site:arxiv.org OR site:paperswithcode.com\n"
            "4. THEN use memory to create shared/research_findings.md with key findings\n"
        )
    else:
        memory_instructions = (
            "\nWORKFLOW:\n"
            "1. FIRST call kaggle_search_competitions() if no target is provided\n"
            "2. THEN call kaggle_get_leaderboard() to analyze current standings\n"
            "3. THEN call web_search() with site:arxiv.org OR site:paperswithcode.com\n"
            "4. THEN summarize findings clearly for the EVOLVER\n"
        )

    instructions = (
        "You are the SCIENTIST agent in the AGENT-K multi-agent system.\n\n"
        "Your mission is to research a competition and develop a winning strategy.\n\n"
        "AVAILABLE TOOLS:\n"
        "- kaggle_get_competition: Get competition details\n"
        "- kaggle_get_leaderboard: Analyze current standings\n"
        "- web_search: Built-in web search\n"
        f"{memory_instructions}"
        "\nAt the end, provide a research summary with recommended approaches.\n"
    )

    return Agent(
        model,
        deps_type=KaggleDeps,
        output_type=str,  # Simple string output for broad compatibility
        toolsets=[kaggle_toolset],
        builtin_tools=builtin_tools,
        instructions=instructions,
        name="scientist",
        retries=5,
    )


# =============================================================================
# Demo Runner
# =============================================================================
async def run_demo(model_spec: str = "anthropic:claude-3-haiku-20240307") -> None:
    """Run the multi-agent demo.

    Args:
        model_spec: The model to use (e.g., 'anthropic:claude-3-haiku-20240307',
                   'openai:gpt-4o')
    """

    _bootstrap()

    from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
    from agent_k.core.deps import KaggleDeps
    from agent_k.toolsets import (
        create_memory_backend,
        kaggle_toolset,
        prepare_memory_tool,
        prepare_web_search,
        register_memory_tool,
    )
    from agent_k.ui.ag_ui import EventEmitter

    console.print(
        Panel.fit(
            "[bold cyan]AGENT-K Multi-Agent Playbook Demo[/bold cyan]\n"
            "[dim]WebSearchTool • MCPServerTool(kaggle) • MemoryTool[/dim]",
            border_style="cyan",
        )
    )

    # Check credentials
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")

    if not username or not api_key:
        console.print("[red]Error:[/red] KAGGLE_USERNAME and KAGGLE_KEY not set")
        return

    console.print(f"\n[green]✓[/green] Model: [bold]{model_spec}[/bold]")
    console.print(f"[green]✓[/green] Kaggle: {username}")

    # Create toolsets and shared dependencies
    memory_dir = Path(__file__).parent / "mission_memory"

    kaggle_config = KaggleSettings(username=username, api_key=api_key)
    kaggle_adapter = KaggleAdapter(kaggle_config)

    kaggle_deps = KaggleDeps(
        event_emitter=EventEmitter(),
        kaggle_adapter=kaggle_adapter,
    )

    memory_backend: Any | None = None
    if model_spec.startswith("anthropic:"):
        try:
            memory_backend = create_memory_backend(storage_path=memory_dir)
        except RuntimeError:
            memory_backend = None

    memory_enabled = memory_backend is not None

    builtin_tools: list[Any] = [prepare_web_search]
    if memory_enabled:
        builtin_tools.append(prepare_memory_tool)

    # Create agents with the specified model
    lobbyist = create_lobbyist_agent(model_spec, kaggle_toolset, builtin_tools, memory_enabled)
    scientist = create_scientist_agent(model_spec, kaggle_toolset, builtin_tools, memory_enabled)

    if memory_enabled:
        assert memory_backend is not None
        register_memory_tool(lobbyist, memory_backend)
        register_memory_tool(scientist, memory_backend)

    async with kaggle_adapter:
        # =========================================================
        # Phase 1: LOBBYIST - Discovery
        # =========================================================
        tools_description = (
            "• kaggle_search_competitions (Kaggle API)\n• web_search (Built-in WebSearchTool)\n"
        )
        if memory_enabled:
            tools_description += "• memory (Built-in MemoryTool)\n"

        console.print(
            Panel(
                "[bold]Phase 1: LOBBYIST - Competition Discovery[/bold]\n\n"
                "Tools being used:\n"
                f"{tools_description}",
                title="🎯 Discovery Phase",
                border_style="blue",
            )
        )

        if memory_enabled:
            discovery_prompt = (
                "Find active Kaggle competitions that would be good for demonstrating ML skills.\n\n"
                "STEPS TO FOLLOW:\n"
                "1. Call kaggle_search_competitions() - find active competitions\n"
                '2. Call web_search(query="Kaggle competition 2025") - find recent news\n'
                "3. Use memory to create shared/target_competition.md with the best competition\n\n"
                "Then summarize what you found.\n"
            )
        else:
            discovery_prompt = (
                "Find active Kaggle competitions that would be good for demonstrating ML skills.\n\n"
                "STEPS TO FOLLOW:\n"
                "1. Call kaggle_search_competitions() - find active competitions\n"
                '2. Call web_search(query="Kaggle competition 2025") - find recent news\n'
                "3. Summarize the best competition clearly for the SCIENTIST\n\n"
                "Then summarize what you found.\n"
            )

        console.print("\n[yellow]LOBBYIST analyzing...[/yellow]\n")

        try:
            result = await lobbyist.run(discovery_prompt, deps=kaggle_deps)
            console.print(Panel(result.output, title="LOBBYIST Results", border_style="blue"))

        except Exception as e:
            console.print(f"[red]LOBBYIST error:[/red] {e}")
            import traceback

            traceback.print_exc()
            # Continue to scientist anyway

        # =========================================================
        # Phase 2: SCIENTIST - Research
        # =========================================================
        tools_description = (
            "• kaggle_get_leaderboard (Kaggle API)\n• web_search (Built-in WebSearchTool)\n"
        )
        if memory_enabled:
            tools_description += "• memory (Built-in MemoryTool)\n"

        console.print(
            Panel(
                "[bold]Phase 2: SCIENTIST - Competition Research[/bold]\n\n"
                "Tools being used:\n"
                f"{tools_description}",
                title="🔬 Research Phase",
                border_style="green",
            )
        )

        if memory_enabled:
            research_prompt = (
                "Research the competition stored by LOBBYIST.\n\n"
                "STEPS TO FOLLOW:\n"
                "1. Use memory to view shared/target_competition.md\n"
                '2. Call kaggle_get_leaderboard(competition_id="<id>") - analyze standings\n'
                '3. Call web_search(query="site:arxiv.org OR site:paperswithcode.com <topic>")\n'
                "4. Use memory to create shared/research_findings.md with key findings\n\n"
                "Then summarize your research and recommended approaches.\n"
            )
        else:
            research_prompt = (
                "Research a promising Kaggle competition.\n\n"
                "STEPS TO FOLLOW:\n"
                "1. Call kaggle_search_competitions() if you need a target\n"
                '2. Call kaggle_get_leaderboard(competition_id="<id>") - analyze standings\n'
                '3. Call web_search(query="site:arxiv.org OR site:paperswithcode.com <topic>")\n'
                "4. Summarize your research and recommended approaches.\n"
            )

        console.print("\n[yellow]SCIENTIST researching...[/yellow]\n")

        try:
            result = await scientist.run(research_prompt, deps=kaggle_deps)
            console.print(Panel(result.output, title="SCIENTIST Results", border_style="green"))

        except Exception as e:
            console.print(f"[red]SCIENTIST error:[/red] {e}")
            import traceback

            traceback.print_exc()

    # =========================================================
    # Summary
    # =========================================================
    memory_summary = ""
    if memory_enabled:
        memory_summary = f"\n• [cyan]MemoryTool[/cyan] - file-backed memory at {memory_dir.name}"

    console.print(
        Panel(
            "[bold green]✓ Multi-Agent Playbook Complete[/bold green]\n\n"
            f"Model used: [bold]{model_spec}[/bold]\n\n"
            "Toolsets demonstrated:\n"
            "• [cyan]KaggleToolset[/cyan] - kaggle_search_competitions, kaggle_get_leaderboard\n"
            "• [cyan]WebSearchTool[/cyan] - web_search (built-in)"
            f"{memory_summary}\n\n"
            "These tools are executed by the model provider when supported.",
            title="📋 Summary",
            border_style="green",
        )
    )

    if memory_enabled and memory_dir.exists():
        files = [p for p in memory_dir.rglob("*") if p.is_file()]
        console.print(f"\n[dim]Memory files: {len(files)}[/dim]")
        for path in files:
            rel = path.relative_to(memory_dir)
            preview = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            snippet = preview[0][:60] if preview else ""
            console.print(f"  • {rel}: {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGENT-K Multi-Agent Playbook Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Claude Haiku (Anthropic)
  python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

  # GPT-4o (OpenAI Responses)
  python examples/multi_agent_playbook.py --model openai:gpt-4o
""",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="anthropic:claude-3-haiku-20240307",
        choices=SUPPORTED_MODELS,
        help="Model to use for agents (default: anthropic:claude-3-haiku-20240307)",
    )

    args = parser.parse_args()
    asyncio.run(run_demo(model_spec=args.model))
