#!/usr/bin/env python3
"""Example: Running an AGENT-K mission with Devstral.

This example demonstrates how to configure and run the AGENT-K multi-agent
system using the local Devstral model instead of Claude.

Prerequisites:
    - LM Studio running locally with Devstral model loaded
    - Server started at http://192.168.105.1:1234/v1 (or configure DEVSTRAL_BASE_URL)
"""
from __future__ import annotations

import asyncio
import os
import sys

# Add the backend directory to the path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_k.agents.lycurgus import LycurgusOrchestrator, OrchestratorConfig
from agent_k.core.models import MissionCriteria, CompetitionType
from agent_k.infra import configure_instrumentation


async def run_devstral_mission():
    """Run a complete AGENT-K mission using Devstral.
    
    This demonstrates the full playbook lifecycle:
    1. Discovery - Find competitions matching criteria
    2. Research - Analyze competition and form strategy
    3. Prototype - Build baseline solution
    4. Evolution - Optimize using evolutionary search
    5. Submission - Submit final solution
    """
    # Configure observability
    configure_instrumentation(service_name='agent-k-devstral')
    
    # Create orchestrator with Devstral model
    config = OrchestratorConfig.with_devstral()
    print(f"Using model: {config.default_model}")
    
    orchestrator = LycurgusOrchestrator(config=config)
    
    # Define mission criteria
    criteria = MissionCriteria(
        target_domains=['tabular', 'classification'],
        target_competition_types=[CompetitionType.PLAYGROUND],
        min_days_remaining=7,
        target_leaderboard_percentile=0.25,  # Target top 25%
        max_evolution_rounds=10,  # Reduced for demo
    )
    
    print("\n" + "=" * 60)
    print("AGENT-K Mission with Devstral")
    print("=" * 60)
    print(f"\nMission Criteria:")
    print(f"  - Domains: {criteria.target_domains}")
    print(f"  - Types: {[t.value for t in criteria.target_competition_types]}")
    print(f"  - Min days remaining: {criteria.min_days_remaining}")
    print(f"  - Target percentile: {criteria.target_leaderboard_percentile}")
    print(f"  - Max evolution rounds: {criteria.max_evolution_rounds}")
    print()
    
    # Execute mission
    async with orchestrator:
        try:
            result = await orchestrator.execute_mission(
                competition_id='playground-tabular-demo',  # Replace with actual competition
                criteria=criteria,
            )
            
            print("\n" + "=" * 60)
            print("Mission Complete!")
            print("=" * 60)
            print(f"\nResult:")
            print(f"  - Success: {result.success}")
            print(f"  - Final Rank: {result.final_rank}")
            print(f"  - Final Score: {result.final_score}")
            print(f"  - Phases Completed: {result.phases_completed}")
            print(f"  - Duration: {result.duration_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"\nMission failed: {e}")
            raise


async def run_simple_agent_test():
    """Run a simple test of a single agent with Devstral.
    
    This is useful for testing the integration without running
    a full mission.
    """
    from dataclasses import dataclass, field
    from typing import Any
    
    import httpx
    
    from agent_k.agents.lobbyist import LobbyistAgent, LobbyistDeps
    from agent_k.core.constants import DEVSTRAL_MODEL
    from agent_k.adapters.kaggle import KaggleAdapter
    from agent_k.ui.ag_ui.event_stream import EventEmitter
    
    print("\n" + "=" * 60)
    print("Simple Agent Test with Devstral")
    print("=" * 60)
    
    # Create agent with Devstral
    agent = LobbyistAgent(model=DEVSTRAL_MODEL)
    print(f"\nCreated Lobbyist agent with Devstral")
    
    # Create mock dependencies
    @dataclass
    class MockEventEmitter:
        async def emit_tool_start(self, **kwargs): pass
        async def emit_tool_result(self, **kwargs): pass
        async def emit(self, event: str, data: dict): pass
    
    @dataclass
    class MockPlatformAdapter:
        async def search_competitions(self, **kwargs):
            return iter([])  # Empty for demo
        async def get_competition(self, competition_id: str):
            return None
    
    async with httpx.AsyncClient() as client:
        deps = LobbyistDeps(
            http_client=client,
            platform_adapter=MockPlatformAdapter(),
            event_emitter=MockEventEmitter(),
        )
        
        # Run discovery
        prompt = """
        Find Kaggle playground competitions suitable for beginners.
        Focus on tabular data classification or regression problems.
        """
        
        print(f"\nRunning discovery with prompt:")
        print(f"  {prompt.strip()}")
        
        try:
            result = await agent.run(prompt, deps=deps)
            print(f"\nDiscovery result:")
            print(f"  - Competitions found: {len(result.competitions)}")
            print(f"  - Total searched: {result.total_searched}")
            print(f"  - Filters applied: {result.filters_applied}")
            print("\n✓ Agent test completed successfully!")
        except Exception as e:
            print(f"\n⚠ Agent test failed: {e}")
            print("  This may be expected if Kaggle MCP is not configured.")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AGENT-K with Devstral')
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Run simple agent test instead of full mission',
    )
    args = parser.parse_args()
    
    if args.simple:
        await run_simple_agent_test()
    else:
        await run_devstral_mission()


if __name__ == '__main__':
    asyncio.run(main())

