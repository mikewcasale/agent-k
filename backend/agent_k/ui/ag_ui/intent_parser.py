"""Intent parser for detecting mission requests from chat messages.

Parses natural language messages to extract Kaggle competition mission criteria.
Uses pydantic-ai for structured output extraction.
"""
from __future__ import annotations

import logfire
from pydantic_ai import Agent

from ...core.models import CompetitionType, MissionCriteria

__all__ = ['parse_mission_intent', 'MissionIntentResult']


# =============================================================================
# Intent Detection Agent
# =============================================================================
INTENT_SYSTEM_PROMPT = """You are an intent detection system for AGENT-K, a multi-agent Kaggle competition system.

Your task is to determine if the user wants to start a Kaggle competition mission, and if so, extract the mission criteria.

A mission request typically includes phrases like:
- "find a competition", "find me a Kaggle competition"
- "compete in", "enter a competition"
- "discover competitions", "search for challenges"
- "participate in Kaggle"

If the user is asking about AGENT-K itself, explaining what it does, or asking for help, that is NOT a mission request.

Extract the following criteria if mentioned:
- Competition types: featured, research, playground, getting_started, community
- Minimum prize pool (in USD)
- Minimum days remaining before deadline
- Target domains: computer vision (cv), natural language processing (nlp), tabular data, time series, etc.
- Target leaderboard percentile (e.g., "top 10%" = 0.10)
- Population size for evolution (default: 50)
- Max evolution rounds (default: 100)

If the message is NOT about starting a mission, return is_mission=False with null criteria.
If it IS a mission request, return is_mission=True with extracted criteria (use defaults for unspecified fields).
"""


class MissionIntentResult:
    """Result of mission intent parsing."""

    def __init__(self, is_mission: bool, criteria: MissionCriteria | None = None):
        self.is_mission = is_mission
        self.criteria = criteria


async def parse_mission_intent(messages: list[dict]) -> MissionCriteria | None:
    """Parse chat messages to detect mission intent and extract criteria.

    Args:
        messages: List of message dicts with 'role' and 'parts' fields.

    Returns:
        MissionCriteria if mission detected, None otherwise.
    """
    # Extract the latest user message
    user_messages = [m for m in messages if m.get('role') == 'user']
    if not user_messages:
        return None

    latest_message = user_messages[-1]

    # Extract text from parts
    text_parts = [
        p.get('text', '')
        for p in latest_message.get('parts', [])
        if p.get('type') == 'text'
    ]
    message_text = ' '.join(text_parts).strip()

    if not message_text:
        return None

    # Check for mission keywords (quick pre-filter)
    mission_keywords = [
        'find', 'competition', 'kaggle', 'compete', 'enter',
        'discover', 'search', 'challenge', 'participate',
    ]

    text_lower = message_text.lower()
    has_keyword = any(keyword in text_lower for keyword in mission_keywords)

    # Also check for anti-keywords that indicate it's NOT a mission
    anti_keywords = ['what is', 'explain', 'how does', 'tell me about', 'help']
    has_anti_keyword = any(anti in text_lower for anti in anti_keywords)

    # Quick reject if no mission keywords or has anti-keywords
    if not has_keyword or has_anti_keyword:
        logfire.debug('intent_parse_rejected', message=message_text[:100], reason='keywords')
        return None

    # Use LLM for deeper intent detection and criteria extraction
    try:
        with logfire.span('parse_mission_intent', message_preview=message_text[:100]):
            # Build prompt for the agent
            prompt = f"""Analyze this user message and determine if they want to start a Kaggle competition mission:

"{message_text}"

Extract mission criteria if this is a mission request."""

            # For now, use simple heuristic parsing until we can properly structure the agent
            # TODO: Use pydantic-ai with structured output once we verify the agent setup

            # Simple heuristic extraction
            criteria_dict = {}

            # Extract competition types
            if 'featured' in text_lower:
                criteria_dict['target_competition_types'] = [CompetitionType.FEATURED]
            elif 'research' in text_lower:
                criteria_dict['target_competition_types'] = [CompetitionType.RESEARCH]
            elif 'playground' in text_lower:
                criteria_dict['target_competition_types'] = [CompetitionType.PLAYGROUND]

            # Extract prize pool (look for $X,XXX,XXX or $Xk/$Xm patterns)
            import re
            # Use word boundary \b after multiplier to avoid matching "m" in "minimum"
            prize_match = re.search(r'\$([\d,]+(?:\.\d+)?)\s*(k|m|thousand|million)?\b', text_lower)
            if prize_match:
                # Remove commas and parse the number
                amount_str = prize_match.group(1).replace(',', '')
                multiplier_suffix = prize_match.group(2)

                # Handle decimal amounts (e.g., "$1.5m")
                amount = float(amount_str)

                if multiplier_suffix in ['k', 'thousand']:
                    amount *= 1000
                elif multiplier_suffix in ['m', 'million']:
                    amount *= 1_000_000

                criteria_dict['min_prize_pool'] = int(amount)

            # Extract days remaining
            days_match = re.search(r'(\d+)\s*(days?|weeks?)', text_lower)
            if days_match:
                days = int(days_match.group(1))
                if 'week' in days_match.group(2):
                    days *= 7
                criteria_dict['min_days_remaining'] = days

            # Extract domains
            domains = []
            if 'computer vision' in text_lower or 'cv' in text_lower or 'image' in text_lower:
                domains.append('computer_vision')
            if 'nlp' in text_lower or 'natural language' in text_lower or 'text' in text_lower:
                domains.append('nlp')
            if 'tabular' in text_lower:
                domains.append('tabular')
            if domains:
                criteria_dict['target_domains'] = domains

            # Extract percentile goal
            percentile_match = re.search(r'top\s+(\d+)%', text_lower)
            if percentile_match:
                percentile = int(percentile_match.group(1)) / 100
                criteria_dict['target_leaderboard_percentile'] = percentile

            # If we extracted any criteria, it's a mission
            if criteria_dict:
                criteria = MissionCriteria(**criteria_dict)
                logfire.info('mission_intent_detected', criteria=criteria_dict)
                return criteria

            # Default: if has mission keywords, create basic mission
            if has_keyword and not has_anti_keyword:
                criteria = MissionCriteria()  # Use defaults
                logfire.info('mission_intent_detected_default')
                return criteria

            return None

    except Exception as e:
        logfire.error('intent_parse_error', error=str(e))
        return None
