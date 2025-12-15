#!/usr/bin/env python3
"""Discover Kaggle competitions using AGENT-K with Devstral.

This example demonstrates the Discovery phase of the AGENT-K playbook
using the local Devstral model.

Usage:
    python examples/devstral_discovery.py

Prerequisites:
    - LM Studio running with Devstral model at http://192.168.105.1:1234
    - KAGGLE_USERNAME and KAGGLE_KEY environment variables set
"""
from __future__ import annotations

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from agent_k.adapters.kaggle import KaggleAdapter, KaggleConfig
from agent_k.agents.lobbyist.agent_devstral import DevstralLobbyistAgent, LobbyistDeps
from agent_k.core.constants import DEVSTRAL_MODEL

console = Console()


async def main():
    """Run competition discovery with Devstral."""
    
    console.print(Panel.fit(
        "[bold cyan]AGENT-K Discovery Phase[/bold cyan]\n"
        "[dim]Using Devstral local model[/dim]",
        border_style="cyan",
    ))
    
    # Check credentials
    username = os.getenv('KAGGLE_USERNAME')
    api_key = os.getenv('KAGGLE_KEY')
    
    if not username or not api_key:
        console.print("[red]Error:[/red] KAGGLE_USERNAME and KAGGLE_KEY not set")
        return
    
    console.print(f"\n[green]✓[/green] Kaggle credentials found: {username}")
    console.print(f"[green]✓[/green] Devstral URL: {os.getenv('DEVSTRAL_BASE_URL', 'http://192.168.105.1:1234/v1')}")
    
    # Create adapter and agent
    config = KaggleConfig(username=username, api_key=api_key)
    adapter = KaggleAdapter(config)
    
    async with adapter:
        # Authenticate
        console.print("\n[yellow]Authenticating with Kaggle API...[/yellow]")
        authenticated = await adapter.authenticate()
        
        if not authenticated:
            console.print("[red]Error:[/red] Failed to authenticate with Kaggle")
            return
        
        console.print("[green]✓[/green] Authenticated successfully\n")
        
        # Create Devstral agent
        console.print("[yellow]Creating LOBBYIST agent with Devstral...[/yellow]")
        agent = DevstralLobbyistAgent(model=DEVSTRAL_MODEL)
        console.print("[green]✓[/green] Agent created\n")
        
        # Run discovery
        console.print(Panel(
            "I want to find Kaggle competitions that:\n"
            "• Have at least 30 days remaining\n"
            "• Have substantial prize pools\n"
            "• Are suitable for advanced ML techniques",
            title="[bold]Mission Criteria[/bold]",
            border_style="blue",
        ))
        
        prompt = """
        Find Kaggle competitions that would be good to compete in.
        
        Use the search_kaggle_competitions tool to get active competitions.
        Then analyze the results and select the top 3 competitions that:
        - Have at least 30 days remaining before deadline
        - Have prize pools > $50,000
        - Use clear evaluation metrics
        
        For each selected competition, call get_competition_details to get
        more information.
        """
        
        console.print("\n[yellow]Running discovery with Devstral...[/yellow]")
        console.print("[dim]This may take a moment while the model thinks...[/dim]\n")
        
        async with httpx.AsyncClient() as client:
            deps = LobbyistDeps(
                http_client=client,
                platform_adapter=adapter,
            )
            
            try:
                result = await agent.run(prompt, deps=deps)
                
                # Display results
                console.print("[green]✓[/green] Discovery complete!\n")
                
                table = Table(title="Discovered Competitions")
                table.add_column("Rank", style="cyan", no_wrap=True)
                table.add_column("Competition", style="green")
                table.add_column("Type", style="magenta")
                table.add_column("Days Left", justify="right")
                table.add_column("Prize", justify="right", style="yellow")
                
                for i, comp in enumerate(result.competitions, 1):
                    prize = f"${comp.prize_pool:,}" if comp.prize_pool else "N/A"
                    table.add_row(
                        str(i),
                        comp.title[:50],
                        comp.competition_type.value,
                        str(comp.days_remaining),
                        prize,
                    )
                
                console.print(table)
                
                console.print(f"\n[dim]Total searched: {result.total_searched}[/dim]")
                console.print(f"[dim]Filters applied: {result.filters_applied}[/dim]")
                
                # Show recommended action
                if result.competitions:
                    top = result.competitions[0]
                    console.print(Panel(
                        f"[bold]Recommended:[/bold] {top.title}\n\n"
                        f"ID: {top.id}\n"
                        f"Type: {top.competition_type.value}\n"
                        f"Metric: {top.metric.value}\n"
                        f"Days remaining: {top.days_remaining}\n"
                        f"Prize: ${top.prize_pool:,}" if top.prize_pool else "N/A",
                        title="[bold green]Next Step: Research Phase[/bold green]",
                        border_style="green",
                    ))
                
            except Exception as e:
                console.print(f"[red]Error during discovery:[/red] {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())

