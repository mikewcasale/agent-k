"""State machine nodes for AGENT-K mission graph.

Each node represents a phase in the competition lifecycle.
Per spec Section 3, nodes follow visibility-based ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import logfire
from pydantic_graph import BaseNode, End, GraphRunContext

from ..core.constants import (
    DISCOVERY_TIMEOUT_SECONDS,
    EVOLUTION_TIMEOUT_SECONDS,
    PROTOTYPE_TIMEOUT_SECONDS,
    RESEARCH_TIMEOUT_SECONDS,
    SUBMISSION_TIMEOUT_SECONDS,
)
from ..core.exceptions import PhaseTimeoutError
from ..core.models import MissionResult, PhasePlan
from .state import GraphContext, MissionState

if TYPE_CHECKING:
    from ..agents.evolver import EvolverAgent
    from ..agents.lobbyist import LobbyistAgent
    from ..agents.scientist import ScientistAgent
    from ..ui.ag_ui.event_stream import EventEmitter

__all__ = [
    'DiscoveryNode',
    'ResearchNode',
    'PrototypeNode',
    'EvolutionNode',
    'SubmissionNode',
]


# =============================================================================
# Discovery Node
# =============================================================================
@dataclass
class DiscoveryNode(BaseNode[MissionState, MissionResult]):
    """Discovery phase node.
    
    Executes the LOBBYIST agent to discover competitions matching criteria.
    
    Transitions:
        - Success → ResearchNode
        - Failure → End(failure)
    """
    
    lobbyist_agent: Any  # LobbyistAgent
    timeout: int = DISCOVERY_TIMEOUT_SECONDS
    
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> ResearchNode | End[MissionResult]:
        """Execute discovery phase."""
        state = ctx.state
        emitter: EventEmitter = ctx.deps.event_emitter
        
        with logfire.span(
            'graph.discovery',
            mission_id=state.mission_id,
        ):
            # Emit phase start
            await emitter.emit_phase_start(
                phase='discovery',
                objectives=[
                    'Find competitions matching criteria',
                    'Validate competition accessibility',
                    'Rank by fit score',
                ],
            )
            
            state.current_phase = 'discovery'
            state.phase_started_at = datetime.now(timezone.utc)
            
            try:
                # Build prompt from criteria
                prompt = self._build_discovery_prompt(state.criteria)
                
                # Create dependencies
                from ..agents.lobbyist import LobbyistDeps
                deps = LobbyistDeps(
                    http_client=ctx.deps.http_client,
                    platform_adapter=ctx.deps.platform_adapter,
                    event_emitter=emitter,
                )
                
                # Run lobbyist agent
                result = await self.lobbyist_agent.run(prompt, deps=deps)
                
                # Update state
                state.discovered_competitions = result.competitions
                
                if not result.competitions:
                    await emitter.emit_phase_complete(
                        phase='discovery',
                        success=False,
                        duration_ms=self._elapsed_ms(state.phase_started_at),
                    )
                    return End(MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message='No competitions found matching criteria',
                    ))
                
                # Select best competition
                state.selected_competition = result.competitions[0]
                state.competition_id = state.selected_competition.id
                state.phases_completed.append('discovery')
                
                await emitter.emit_phase_complete(
                    phase='discovery',
                    success=True,
                    duration_ms=self._elapsed_ms(state.phase_started_at),
                )
                
                # Transition to research
                return ResearchNode(scientist_agent=self._get_scientist_agent())
                
            except Exception as e:
                logfire.error('discovery_failed', error=str(e))
                await emitter.emit_error(
                    error_id=f'discovery_{state.mission_id}',
                    category='recoverable',
                    error_type=type(e).__name__,
                    message=str(e),
                    context='Discovery phase',
                    recovery_strategy='retry',
                )
                return End(MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message=f'Discovery failed: {e}',
                ))
    
    def _build_discovery_prompt(self, criteria: Any) -> str:
        """Build discovery prompt from criteria."""
        parts = ['Find Kaggle competitions with the following criteria:']
        
        if criteria.target_competition_types:
            types = ', '.join(t.value for t in criteria.target_competition_types)
            parts.append(f'- Types: {types}')
        
        if criteria.min_prize_pool:
            parts.append(f'- Minimum prize: ${criteria.min_prize_pool:,}')
        
        if criteria.min_days_remaining:
            parts.append(f'- At least {criteria.min_days_remaining} days remaining')
        
        if criteria.target_domains:
            domains = ', '.join(criteria.target_domains)
            parts.append(f'- Domains: {domains}')
        
        parts.append(f'- Target top {criteria.target_leaderboard_percentile*100:.0f}% on leaderboard')
        
        return '\n'.join(parts)
    
    def _elapsed_ms(self, start: datetime | None) -> int:
        """Calculate elapsed milliseconds."""
        if not start:
            return 0
        delta = datetime.now(timezone.utc) - start
        return int(delta.total_seconds() * 1000)
    
    def _get_scientist_agent(self) -> Any:
        """Get scientist agent for next phase."""
        from ..agents.scientist import ScientistAgent
        return ScientistAgent()


# =============================================================================
# Research Node
# =============================================================================
@dataclass
class ResearchNode(BaseNode[MissionState, MissionResult]):
    """Research phase node.
    
    Executes the SCIENTIST agent to analyze the competition.
    
    Transitions:
        - Success → PrototypeNode
        - Failure → End(failure)
    """
    
    scientist_agent: Any  # ScientistAgent
    timeout: int = RESEARCH_TIMEOUT_SECONDS
    
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> PrototypeNode | End[MissionResult]:
        """Execute research phase."""
        state = ctx.state
        emitter: EventEmitter = ctx.deps.event_emitter
        
        with logfire.span(
            'graph.research',
            competition_id=state.competition_id,
        ):
            await emitter.emit_phase_start(
                phase='research',
                objectives=[
                    'Analyze leaderboard and score distribution',
                    'Review relevant papers and techniques',
                    'Perform exploratory data analysis',
                    'Synthesize strategy recommendations',
                ],
            )
            
            state.current_phase = 'research'
            state.phase_started_at = datetime.now(timezone.utc)
            
            try:
                # Research implementation
                from ..agents.scientist import ScientistDeps
                deps = ScientistDeps(
                    http_client=ctx.deps.http_client,
                    platform_adapter=ctx.deps.platform_adapter,
                    competition=state.selected_competition,
                )
                
                prompt = f"Research competition: {state.selected_competition.title}"
                result = await self.scientist_agent.research(prompt, deps=deps)
                
                state.research_findings = result
                state.phases_completed.append('research')
                
                await emitter.emit_phase_complete(
                    phase='research',
                    success=True,
                    duration_ms=self._elapsed_ms(state.phase_started_at),
                )
                
                return PrototypeNode()
                
            except Exception as e:
                logfire.error('research_failed', error=str(e))
                return End(MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message=f'Research failed: {e}',
                    phases_completed=list(state.phases_completed),
                ))
    
    def _elapsed_ms(self, start: datetime | None) -> int:
        if not start:
            return 0
        delta = datetime.now(timezone.utc) - start
        return int(delta.total_seconds() * 1000)


# =============================================================================
# Prototype Node
# =============================================================================
@dataclass
class PrototypeNode(BaseNode[MissionState, MissionResult]):
    """Prototype phase node.
    
    Generates initial baseline solution.
    
    Transitions:
        - Success → EvolutionNode
        - Failure → End(failure)
    """
    
    timeout: int = PROTOTYPE_TIMEOUT_SECONDS
    
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> EvolutionNode | End[MissionResult]:
        """Execute prototype phase."""
        state = ctx.state
        emitter: EventEmitter = ctx.deps.event_emitter
        
        with logfire.span('graph.prototype', competition_id=state.competition_id):
            await emitter.emit_phase_start(
                phase='prototype',
                objectives=[
                    'Generate baseline solution code',
                    'Validate solution structure',
                    'Establish baseline score',
                ],
            )
            
            state.current_phase = 'prototype'
            state.phase_started_at = datetime.now(timezone.utc)
            
            try:
                # Generate prototype based on research findings
                # This would use CodeExecutionTool to validate
                state.prototype_code = self._generate_prototype(
                    state.selected_competition,
                    state.research_findings,
                )
                
                state.phases_completed.append('prototype')
                
                await emitter.emit_phase_complete(
                    phase='prototype',
                    success=True,
                    duration_ms=self._elapsed_ms(state.phase_started_at),
                )
                
                return EvolutionNode(evolver_agent=self._get_evolver_agent())
                
            except Exception as e:
                logfire.error('prototype_failed', error=str(e))
                return End(MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message=f'Prototype failed: {e}',
                    phases_completed=list(state.phases_completed),
                ))
    
    def _generate_prototype(self, competition: Any, research: Any) -> str:
        """Generate prototype solution code."""
        # Implementation would generate baseline code
        return "# Prototype solution\n"
    
    def _elapsed_ms(self, start: datetime | None) -> int:
        if not start:
            return 0
        delta = datetime.now(timezone.utc) - start
        return int(delta.total_seconds() * 1000)
    
    def _get_evolver_agent(self) -> Any:
        from ..agents.evolver import EvolverAgent
        return EvolverAgent()


# =============================================================================
# Evolution Node
# =============================================================================
@dataclass
class EvolutionNode(BaseNode[MissionState, MissionResult]):
    """Evolution phase node.
    
    Executes the EVOLVER agent to optimize the solution.
    
    Transitions:
        - Success → SubmissionNode
        - Failure → End(failure with best solution)
    """
    
    evolver_agent: Any  # EvolverAgent
    timeout: int = EVOLUTION_TIMEOUT_SECONDS
    
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> SubmissionNode | End[MissionResult]:
        """Execute evolution phase."""
        state = ctx.state
        emitter: EventEmitter = ctx.deps.event_emitter
        
        with logfire.span('graph.evolution', competition_id=state.competition_id):
            await emitter.emit_phase_start(
                phase='evolution',
                objectives=[
                    'Initialize population from prototype',
                    'Evolve solutions over generations',
                    'Track fitness improvements',
                    'Achieve target score or convergence',
                ],
            )
            
            state.current_phase = 'evolution'
            state.phase_started_at = datetime.now(timezone.utc)
            
            try:
                from ..agents.evolver import EvolverDeps
                from ..core.models import EvolutionState
                
                # Initialize evolution state
                state.evolution_state = EvolutionState()
                
                deps = EvolverDeps(
                    competition=state.selected_competition,
                    event_emitter=emitter,
                    initial_solution=state.prototype_code or '',
                    target_score=self._calculate_target_score(state),
                )
                
                prompt = f"""
                Evolve solution for {state.selected_competition.title}.
                Target: Top {state.criteria.target_leaderboard_percentile*100:.0f}% on leaderboard.
                Research suggests: {state.research_findings.strategy_recommendations if state.research_findings else 'N/A'}
                """
                
                result = await self.evolver_agent.evolve(
                    prompt,
                    initial_solution=state.prototype_code or '',
                    deps=deps,
                )
                
                # Update state with evolution results
                state.evolution_state.best_solution = {
                    'code': result.best_solution,
                    'fitness': result.best_fitness,
                }
                state.evolution_state.convergence_detected = result.convergence_achieved
                state.evolution_state.convergence_reason = result.convergence_reason
                
                state.phases_completed.append('evolution')
                
                await emitter.emit_phase_complete(
                    phase='evolution',
                    success=True,
                    duration_ms=self._elapsed_ms(state.phase_started_at),
                )
                
                return SubmissionNode()
                
            except Exception as e:
                logfire.error('evolution_failed', error=str(e))
                # Even on failure, try to submit best solution if available
                if state.evolution_state and state.evolution_state.best_solution:
                    return SubmissionNode()
                return End(MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message=f'Evolution failed: {e}',
                    phases_completed=list(state.phases_completed),
                ))
    
    def _calculate_target_score(self, state: MissionState) -> float:
        """Calculate target score from research findings."""
        if state.research_findings and state.research_findings.leaderboard_analysis:
            return state.research_findings.leaderboard_analysis.target_score
        return 0.0
    
    def _elapsed_ms(self, start: datetime | None) -> int:
        if not start:
            return 0
        delta = datetime.now(timezone.utc) - start
        return int(delta.total_seconds() * 1000)


# =============================================================================
# Submission Node
# =============================================================================
@dataclass
class SubmissionNode(BaseNode[MissionState, MissionResult]):
    """Submission phase node.
    
    Final submission of best solution.
    
    Transitions:
        - Success → End(success)
        - Failure → End(failure)
    """
    
    timeout: int = SUBMISSION_TIMEOUT_SECONDS
    
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> End[MissionResult]:
        """Execute submission phase."""
        state = ctx.state
        emitter: EventEmitter = ctx.deps.event_emitter
        
        with logfire.span('graph.submission', competition_id=state.competition_id):
            await emitter.emit_phase_start(
                phase='submission',
                objectives=[
                    'Generate final predictions',
                    'Submit to Kaggle',
                    'Retrieve final score and rank',
                ],
            )
            
            state.current_phase = 'submission'
            state.phase_started_at = datetime.now(timezone.utc)
            
            try:
                # Get best solution
                best_code = ''
                if state.evolution_state and state.evolution_state.best_solution:
                    best_code = state.evolution_state.best_solution.get('code', '')
                elif state.prototype_code:
                    best_code = state.prototype_code
                
                if not best_code:
                    return End(MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message='No solution available for submission',
                        phases_completed=list(state.phases_completed),
                    ))
                
                # Submit via platform adapter
                submission = await ctx.deps.platform_adapter.submit(
                    state.competition_id,
                    best_code,
                    message=f'AGENT-K mission {state.mission_id}',
                )
                
                state.final_submission_id = submission.id
                
                # Wait for score
                import asyncio
                for _ in range(10):  # Poll for score
                    await asyncio.sleep(5)
                    status = await ctx.deps.platform_adapter.get_submission_status(
                        state.competition_id,
                        submission.id,
                    )
                    if status.public_score is not None:
                        state.final_score = status.public_score
                        break
                
                # Get rank
                leaderboard = await ctx.deps.platform_adapter.get_leaderboard(
                    state.competition_id,
                    limit=10000,
                )
                for entry in leaderboard:
                    if entry.score == state.final_score:
                        state.final_rank = entry.rank
                        break
                
                state.phases_completed.append('submission')
                
                # Emit submission result
                await emitter.emit_submission_result(
                    submission_id=submission.id,
                    generation=len(state.evolution_state.generation_history) if state.evolution_state else 0,
                    cv_score=state.evolution_state.best_solution.get('fitness', 0) if state.evolution_state and state.evolution_state.best_solution else 0,
                    public_score=state.final_score,
                    rank=state.final_rank,
                    total_teams=len(leaderboard),
                )
                
                await emitter.emit_phase_complete(
                    phase='submission',
                    success=True,
                    duration_ms=self._elapsed_ms(state.phase_started_at),
                )
                
                # Calculate total duration
                total_duration_ms = int(
                    (datetime.now(timezone.utc) - state.started_at).total_seconds() * 1000
                )
                
                return End(MissionResult(
                    success=True,
                    mission_id=state.mission_id,
                    competition_id=state.competition_id,
                    final_rank=state.final_rank,
                    final_score=state.final_score,
                    total_submissions=len(state.evolution_state.leaderboard_submissions) if state.evolution_state else 1,
                    evolution_generations=len(state.evolution_state.generation_history) if state.evolution_state else 0,
                    duration_ms=total_duration_ms,
                    phases_completed=list(state.phases_completed),
                ))
                
            except Exception as e:
                logfire.error('submission_failed', error=str(e))
                return End(MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    competition_id=state.competition_id,
                    error_message=f'Submission failed: {e}',
                    phases_completed=list(state.phases_completed),
                ))
    
    def _elapsed_ms(self, start: datetime | None) -> int:
        if not start:
            return 0
        delta = datetime.now(timezone.utc) - start
        return int(delta.total_seconds() * 1000)
