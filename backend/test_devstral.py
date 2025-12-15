#!/usr/bin/env python3
"""Test script for Devstral integration with AGENT-K.

This script verifies that the Devstral model can be used with the AGENT-K agents.
Run with: python -m test_devstral
"""
from __future__ import annotations

import asyncio
import os
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_k.infra.models import (
    DEVSTRAL_BASE_URL,
    DEVSTRAL_MODEL_ID,
    create_devstral_model,
    get_model,
    is_devstral_model,
)


def test_model_resolution():
    """Test that model resolution works correctly."""
    print("Testing model resolution...")
    
    # Test Devstral model detection
    assert is_devstral_model('devstral:local'), "Should detect devstral:local"
    assert is_devstral_model('devstral:http://localhost:1234/v1'), "Should detect custom URL"
    assert not is_devstral_model('anthropic:claude-sonnet-4-5'), "Should not detect anthropic"
    print("  ✓ Devstral model detection works")
    
    # Test model resolution
    resolved = get_model('devstral:local')
    assert resolved is not None, "Should resolve devstral:local"
    print(f"  ✓ Resolved devstral:local to: {type(resolved).__name__}")
    
    # Test standard model passthrough
    standard = get_model('anthropic:claude-sonnet-4-5')
    assert standard == 'anthropic:claude-sonnet-4-5', "Should pass through standard models"
    print("  ✓ Standard model passthrough works")
    
    print("✓ Model resolution tests passed!\n")


def test_devstral_model_creation():
    """Test that Devstral model can be created."""
    print("Testing Devstral model creation...")
    
    model = create_devstral_model()
    print(f"  ✓ Created model: {type(model).__name__}")
    print(f"  ✓ Model ID: {DEVSTRAL_MODEL_ID}")
    print(f"  ✓ Base URL: {DEVSTRAL_BASE_URL}")
    
    print("✓ Model creation tests passed!\n")


async def test_devstral_connection():
    """Test that we can connect to the Devstral server."""
    print("Testing Devstral connection...")
    
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            # Check if the server is reachable
            response = await client.get(
                f"{DEVSTRAL_BASE_URL.rstrip('/v1')}/models",
                timeout=5.0,
            )
            if response.status_code == 200:
                models = response.json()
                print(f"  ✓ Server is reachable")
                print(f"  ✓ Available models: {[m.get('id') for m in models.get('data', [])]}")
            else:
                print(f"  ⚠ Server responded with status: {response.status_code}")
    except httpx.ConnectError:
        print(f"  ⚠ Could not connect to {DEVSTRAL_BASE_URL}")
        print("    Make sure LM Studio is running and the server is started")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print("Connection test completed.\n")


async def test_agent_with_devstral():
    """Test that an agent can use Devstral."""
    print("Testing agent with Devstral...")
    
    from pydantic_ai import Agent
    
    # Create a simple agent with Devstral
    model = create_devstral_model()
    agent = Agent(
        model,
        instructions="You are a helpful assistant. Respond concisely.",
    )
    
    print(f"  ✓ Created agent with Devstral model")
    
    try:
        result = await agent.run("Say hello in one word.")
        print(f"  ✓ Agent response: {result.output[:100]}...")
        print("✓ Agent test passed!\n")
    except Exception as e:
        print(f"  ⚠ Agent test failed: {e}")
        print("    Make sure LM Studio is running with Devstral loaded\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("AGENT-K Devstral Integration Tests")
    print("=" * 60)
    print()
    
    test_model_resolution()
    test_devstral_model_creation()
    await test_devstral_connection()
    await test_agent_with_devstral()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

