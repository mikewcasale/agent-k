#!/usr/bin/env python3
"""AGENT-K Multi-Agent Playbook Demo.

This demonstrates the full AGENT-K multi-agent flow using all three
core toolsets from the playbook:

1. WebSearchTool - Web search via SearchToolset
2. MCPServerTool(kaggle) - Kaggle API via KaggleToolset  
3. MemoryTool - Persistent memory via MemoryToolset

The demo runs through:
- LOBBYIST: Competition discovery using web search + Kaggle API
- SCIENTIST: Research phase using web search + papers + memory
- Memory persistence across agents

Usage:
    # Local Devstral (LM Studio)
    python examples/multi_agent_playbook.py --model devstral:local
    
    # Claude Haiku (Anthropic)
    python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307
    
    # Devstral via OpenRouter
    python examples/multi_agent_playbook.py --model openrouter:mistralai/devstral-small-2505
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_k.adapters.kaggle import KaggleAdapter, KaggleConfig
from agent_k.infra.models import get_model
from agent_k.toolsets import (
    create_kaggle_toolset,
    create_memory_toolset,
    create_search_toolset,
)

console = Console()

# Supported models for the demo
SUPPORTED_MODELS = [
    'devstral:local',                           # Local LM Studio
    'anthropic:claude-3-haiku-20240307',        # Claude Haiku
    'openrouter:mistralai/devstral-small',      # Devstral Small via OpenRouter (supports tools)
    'openrouter:mistralai/devstral-2512:free',  # Devstral Free via OpenRouter (supports tools)
]


# =============================================================================
# Agent Definitions - Use str output for broad model compatibility
# =============================================================================
def create_lobbyist_agent(
    model_spec: str,
    kaggle_toolset,
    search_toolset,
    memory_toolset,
) -> Agent[None, str]:
    """Create the LOBBYIST agent for competition discovery."""
    
    model = get_model(model_spec)
    
    return Agent(
        model,
        output_type=str,  # Simple string output for broad compatibility
        toolsets=[kaggle_toolset, search_toolset, memory_toolset],
        instructions="""You are the LOBBYIST agent in the AGENT-K multi-agent system.

Your mission is to discover Kaggle competitions that match the user's criteria.

AVAILABLE TOOLS:
- kaggle_search_competitions: Search Kaggle API for competitions
- kaggle_get_competition: Get details about a specific competition
- web_search: Search the web for information
- search_kaggle: Search kaggle.com for discussions
- memory_store: Save important findings for other agents

WORKFLOW:
1. FIRST call kaggle_search_competitions() to find active competitions
2. THEN call web_search() to find recent Kaggle news
3. THEN call memory_store() to save the best competition for the SCIENTIST

At the end, provide a summary of your findings.
""",
        name='lobbyist',
        retries=5,
    )


def create_scientist_agent(
    model_spec: str,
    kaggle_toolset,
    search_toolset,
    memory_toolset,
) -> Agent[None, str]:
    """Create the SCIENTIST agent for research."""
    
    model = get_model(model_spec)
    
    return Agent(
        model,
        output_type=str,  # Simple string output for broad compatibility
        toolsets=[kaggle_toolset, search_toolset, memory_toolset],
        instructions="""You are the SCIENTIST agent in the AGENT-K multi-agent system.

Your mission is to research a competition and develop a winning strategy.

AVAILABLE TOOLS:
- kaggle_get_competition: Get competition details
- kaggle_get_leaderboard: Analyze current standings
- search_papers: Find academic papers on relevant topics
- search_kaggle: Find winning solutions from similar competitions
- memory_retrieve: Get information saved by LOBBYIST
- memory_store: Save your research findings

WORKFLOW:
1. FIRST call memory_retrieve(key="target_competition") to get LOBBYIST's findings
2. THEN call kaggle_get_leaderboard() to analyze current standings
3. THEN call search_papers() to find relevant research
4. THEN call memory_store() to save your key findings

At the end, provide a research summary with recommended approaches.
""",
        name='scientist',
        retries=5,
    )


# =============================================================================
# Demo Runner
# =============================================================================
async def run_demo(model_spec: str = 'devstral:local'):
    """Run the multi-agent demo.
    
    Args:
        model_spec: The model to use (e.g., 'devstral:local', 
                   'anthropic:claude-3-haiku-20240307',
                   'openrouter:mistralai/devstral-small-2505')
    """
    
    console.print(Panel.fit(
        "[bold cyan]AGENT-K Multi-Agent Playbook Demo[/bold cyan]\n"
        "[dim]WebSearchTool • MCPServerTool(kaggle) • MemoryTool[/dim]",
        border_style="cyan",
    ))
    
    # Check credentials
    username = os.getenv('KAGGLE_USERNAME')
    api_key = os.getenv('KAGGLE_KEY')
    
    if not username or not api_key:
        console.print("[red]Error:[/red] KAGGLE_USERNAME and KAGGLE_KEY not set")
        return
    
    console.print(f"\n[green]✓[/green] Model: [bold]{model_spec}[/bold]")
    console.print(f"[green]✓[/green] Kaggle: {username}")
    
    # Create toolsets using factory functions
    memory_path = Path(__file__).parent / 'mission_memory.json'
    
    kaggle_config = KaggleConfig(username=username, api_key=api_key)
    kaggle_adapter = KaggleAdapter(kaggle_config)
    
    kaggle_toolset = create_kaggle_toolset(kaggle_adapter)
    search_toolset = create_search_toolset()
    memory_toolset = create_memory_toolset(storage_path=memory_path)
    
    # Create agents with the specified model
    lobbyist = create_lobbyist_agent(model_spec, kaggle_toolset, search_toolset, memory_toolset)
    scientist = create_scientist_agent(model_spec, kaggle_toolset, search_toolset, memory_toolset)
    
    async with kaggle_adapter:
        # =========================================================
        # Phase 1: LOBBYIST - Discovery
        # =========================================================
        console.print(Panel(
            "[bold]Phase 1: LOBBYIST - Competition Discovery[/bold]\n\n"
            "Tools being used:\n"
            "• kaggle_search_competitions (Kaggle API)\n"
            "• web_search (DuckDuckGo)\n"
            "• memory_store (Persistence)",
            title="🎯 Discovery Phase",
            border_style="blue",
        ))
        
        discovery_prompt = """
Find active Kaggle competitions that would be good for demonstrating ML skills.

STEPS TO FOLLOW:
1. Call kaggle_search_competitions() - find active competitions 
2. Call web_search(query="Kaggle competition 2025") - find recent news
3. Call memory_store(key="target_competition", value=<best competition info>) - save for SCIENTIST

Then summarize what you found.
"""
        
        console.print("\n[yellow]LOBBYIST analyzing...[/yellow]\n")
        
        try:
            result = await lobbyist.run(discovery_prompt)
            console.print(Panel(result.output, title="LOBBYIST Results", border_style="blue"))
            
        except Exception as e:
            console.print(f"[red]LOBBYIST error:[/red] {e}")
            import traceback
            traceback.print_exc()
            # Continue to scientist anyway
        
        # =========================================================
        # Phase 2: SCIENTIST - Research
        # =========================================================
        console.print(Panel(
            "[bold]Phase 2: SCIENTIST - Competition Research[/bold]\n\n"
            "Tools being used:\n"
            "• kaggle_get_leaderboard (Kaggle API)\n"
            "• search_papers (Academic search)\n"
            "• search_kaggle (Kaggle discussions)\n"
            "• memory_retrieve/store (Persistence)",
            title="🔬 Research Phase",
            border_style="green",
        ))
        
        research_prompt = """
Research the competition stored by LOBBYIST.

STEPS TO FOLLOW:
1. Call memory_retrieve(key="target_competition") - get LOBBYIST's findings
2. Call kaggle_search_competitions() if no memory found
3. Call kaggle_get_leaderboard(competition_id="<id>") - analyze standings
4. Call search_papers(topic="machine learning") - find relevant research
5. Call memory_store(key="research_findings", value=<your findings>) - save for EVOLVER

Then summarize your research and recommended approaches.
"""
        
        console.print("\n[yellow]SCIENTIST researching...[/yellow]\n")
        
        try:
            result = await scientist.run(research_prompt)
            console.print(Panel(result.output, title="SCIENTIST Results", border_style="green"))
            
        except Exception as e:
            console.print(f"[red]SCIENTIST error:[/red] {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================
    # Summary
    # =========================================================
    console.print(Panel(
        "[bold green]✓ Multi-Agent Playbook Complete[/bold green]\n\n"
        f"Model used: [bold]{model_spec}[/bold]\n\n"
        "Toolsets demonstrated:\n"
        "• [cyan]KaggleToolset[/cyan] - kaggle_search_competitions, kaggle_get_leaderboard\n"
        "• [cyan]SearchToolset[/cyan] - web_search, search_papers, search_kaggle\n"
        "• [cyan]MemoryToolset[/cyan] - memory_store, memory_retrieve\n\n"
        f"Memory persisted to: {memory_path.name}\n\n"
        "These toolsets work with ANY model because they're executed\n"
        "client-side by pydantic-ai's FunctionToolset.",
        title="📋 Summary",
        border_style="green",
    ))
    
    # Show memory contents
    if memory_path.exists():
        import json
        memory_data = json.loads(memory_path.read_text())
        console.print(f"\n[dim]Memory entries: {len(memory_data.get('entries', []))}[/dim]")
        for entry in memory_data.get('entries', []):
            console.print(f"  • {entry['key']}: {str(entry['value'])[:60]}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AGENT-K Multi-Agent Playbook Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local Devstral (LM Studio)
  python examples/multi_agent_playbook.py --model devstral:local
  
  # Claude Haiku (Anthropic)
  python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307
  
  # Devstral via OpenRouter
  python examples/multi_agent_playbook.py --model openrouter:mistralai/devstral-small-2505
"""
    )
    parser.add_argument(
        '--model', '-m',
        default='devstral:local',
        choices=SUPPORTED_MODELS,
        help='Model to use for agents (default: devstral:local)',
    )
    
    args = parser.parse_args()
    asyncio.run(run_demo(model_spec=args.model))
