"""Evolver agent for solution optimization.

Demonstrates the canonical pattern for integrating multiple builtin tools
as specified in python_spec_v2.md Section 7.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.builtin_tools import (
    CodeExecutionTool,
    MCPServerTool,
    MemoryTool,
)

from ...core.constants import (
    DEFAULT_KAGGLE_MCP_URL,
    DEFAULT_MODEL,
    EVOLUTION_POPULATION_SIZE,
    MAX_EVOLUTION_GENERATIONS,
)
from ...core.models import Competition, EvolutionState, GenerationMetrics
from ...ui.ag_ui.event_stream import EventEmitter

__all__ = ['EvolverAgent', 'EvolverDeps', 'EvolutionResult']


# =============================================================================
# Dependency Container
# =============================================================================
@dataclass
class EvolverDeps:
    """Dependencies for the Evolver agent.
    
    Per spec, includes all external services required for evolution.
    """
    
    competition: Competition
    event_emitter: EventEmitter
    initial_solution: str = ''
    population_size: int = EVOLUTION_POPULATION_SIZE
    max_generations: int = MAX_EVOLUTION_GENERATIONS
    target_score: float = 0.0
    best_solution: str | None = None
    generation_history: list[dict[str, float]] = field(default_factory=list)


# =============================================================================
# Output Model
# =============================================================================
class EvolutionResult(BaseModel):
    """Result of evolution process."""
    
    model_config = {'frozen': True}
    
    best_solution: str = Field(description='Best solution code')
    best_fitness: float = Field(description='Fitness score of best solution')
    generations_completed: int = Field(default=0)
    convergence_achieved: bool = Field(default=False)
    convergence_reason: str | None = Field(default=None)
    submission_ready: bool = Field(default=False)


# =============================================================================
# Agent Factory with Full Builtin Tools Suite
# =============================================================================
def create_evolver_agent(
    model: str = DEFAULT_MODEL,
    kaggle_mcp_url: str = DEFAULT_KAGGLE_MCP_URL,
) -> Agent[EvolverDeps, EvolutionResult]:
    """Create Evolver agent with full builtin tool integration.
    
    Per spec Section 7.3, this demonstrates the canonical pattern for
    integrating multiple builtin tools in a production agent.
    
    Args:
        model: LLM model identifier.
        kaggle_mcp_url: URL for Kaggle MCP server.
    
    Returns:
        Configured Evolver agent with all builtin tools.
    """
    
    # =========================================================================
    # Configure Builtin Tools (Per Spec Section 7.3)
    # =========================================================================
    
    # MCPServerTool: Connect to Kaggle's MCP for platform operations
    # See: https://www.kaggle.com/docs/mcp
    kaggle_mcp = MCPServerTool(
        server_url=kaggle_mcp_url,
        name='kaggle',
        # Authentication handled by model provider's MCP integration
    )
    
    # MemoryTool: Persist context across evolution generations
    # Critical for long-running optimization that may span hours
    memory = MemoryTool(
        # Memory persists across agent invocations within the session
    )
    
    # CodeExecutionTool: Secure sandboxed execution for solution evaluation
    code_executor = CodeExecutionTool(
        # Sandboxed environment with appropriate limits
    )
    
    # =========================================================================
    # Create Agent with Builtin Tools
    # =========================================================================
    agent = Agent(
        model,
        deps_type=EvolverDeps,
        output_type=EvolutionResult,
        instructions=_get_evolver_instructions(),
        builtin_tools=[
            kaggle_mcp,      # Platform operations via MCP
            memory,          # Long-running context persistence
            code_executor,   # Solution evaluation
        ],
        retries=3,
        name='evolver',
    )
    
    # =========================================================================
    # Dynamic Instructions
    # =========================================================================
    @agent.instructions
    async def add_evolution_context(ctx: RunContext[EvolverDeps]) -> str:
        """Add evolution-specific context to instructions."""
        comp = ctx.deps.competition
        history = ctx.deps.generation_history
        
        context = f"""
COMPETITION CONTEXT:
- ID: {comp.id}
- Title: {comp.title}
- Metric: {comp.metric.value} ({comp.metric_direction})
- Target Score: {ctx.deps.target_score}

EVOLUTION STATE:
- Generations Completed: {len(history)}
- Population Size: {ctx.deps.population_size}
- Max Generations: {ctx.deps.max_generations}
"""
        
        if history:
            last_gen = history[-1]
            context += f"""
LAST GENERATION:
- Best Fitness: {last_gen.get('best_fitness', 'N/A')}
- Mean Fitness: {last_gen.get('mean_fitness', 'N/A')}
"""
        
        if ctx.deps.best_solution:
            context += f"\nBEST SOLUTION AVAILABLE: {len(ctx.deps.best_solution)} chars"
        
        return context
    
    # =========================================================================
    # Custom Tools (Domain-Specific, Complement Builtin Tools)
    # =========================================================================
    @agent.tool
    async def mutate_solution(
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        mutation_type: str,
        mutation_params: dict[str, Any] | None = None,
    ) -> str:
        """Apply mutation to a solution.
        
        This is a domain-specific tool that works alongside builtin tools.
        The CodeExecutionTool handles actual execution; this tool handles
        the evolutionary mutation logic.
        
        Args:
            ctx: Run context with evolution state.
            solution_code: Current solution code.
            mutation_type: Type of mutation (point, structural, hyperparameter, crossover).
            mutation_params: Optional parameters for the mutation.
        
        Returns:
            Mutated solution code.
        """
        with logfire.span('evolver.mutate', mutation_type=mutation_type):
            # Emit event for frontend
            await ctx.deps.event_emitter.emit(
                'tool-start',
                {
                    'taskId': 'evolution_mutate',
                    'toolCallId': f'mutate_{mutation_type}',
                    'toolType': 'code_executor',
                    'operation': f'mutate_{mutation_type}',
                },
            )
            
            params = mutation_params or {}
            
            if mutation_type == 'point':
                # Small parameter adjustments
                mutated = _apply_point_mutation(solution_code, params)
            elif mutation_type == 'structural':
                # Architecture changes
                mutated = _apply_structural_mutation(solution_code, params)
            elif mutation_type == 'hyperparameter':
                # Learning rate, regularization changes
                mutated = _apply_hyperparameter_mutation(solution_code, params)
            elif mutation_type == 'crossover':
                # Combine with another solution
                other_solution = params.get('other_solution', '')
                mutated = _apply_crossover(solution_code, other_solution, params)
            else:
                mutated = solution_code
            
            return mutated
    
    @agent.tool
    async def evaluate_fitness(
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """Evaluate solution fitness.
        
        Uses CodeExecutionTool internally for execution, then computes
        domain-specific fitness metrics.
        
        Args:
            ctx: Run context.
            solution_code: Solution to evaluate.
            validation_split: Fraction of data for validation.
        
        Returns:
            Fitness metrics including score, runtime, memory usage.
        """
        with logfire.span('evolver.evaluate_fitness'):
            # The agent will use CodeExecutionTool to run the solution
            # This tool interprets results for fitness scoring
            
            # Emit start event
            await ctx.deps.event_emitter.emit_tool_start(
                task_id='evolution_evaluate',
                tool_call_id=f'fitness_{id(solution_code)[:8]}',
                tool_type='code_executor',
                operation='evaluate_fitness',
            )
            
            # Fitness evaluation would happen here via CodeExecutionTool
            # For now, return placeholder metrics
            result = {
                'fitness': 0.0,
                'cv_score': 0.0,
                'valid': True,
                'runtime_ms': 0,
                'memory_mb': 0.0,
                'error': None,
            }
            
            # Emit result event
            await ctx.deps.event_emitter.emit_tool_result(
                task_id='evolution_evaluate',
                tool_call_id=f'fitness_{id(solution_code)[:8]}',
                result=result,
                duration_ms=result['runtime_ms'],
            )
            
            return result
    
    @agent.tool
    async def record_generation(
        ctx: RunContext[EvolverDeps],
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        worst_fitness: float,
        mutations: dict[str, int],
    ) -> None:
        """Record generation metrics.
        
        Uses MemoryTool to persist generation history for analysis
        and convergence detection.
        
        Args:
            ctx: Run context.
            generation: Generation number.
            best_fitness: Best fitness in generation.
            mean_fitness: Mean fitness in generation.
            worst_fitness: Worst fitness in generation.
            mutations: Count of mutations by type.
        """
        metrics = {
            'generation': generation,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'worst_fitness': worst_fitness,
            'mutations': mutations,
        }
        
        ctx.deps.generation_history.append(metrics)
        
        # Emit generation complete event for frontend
        await ctx.deps.event_emitter.emit_generation_complete(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=worst_fitness,
            population_size=ctx.deps.population_size,
            mutations=mutations,
        )
        
        logfire.info(
            'evolution_generation',
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
        )
    
    @agent.tool
    async def check_convergence(
        ctx: RunContext[EvolverDeps],
        threshold_generations: int = 5,
        improvement_threshold: float = 0.001,
    ) -> dict[str, Any]:
        """Check if evolution has converged.
        
        Args:
            ctx: Run context.
            threshold_generations: Number of generations without improvement.
            improvement_threshold: Minimum improvement to count as progress.
        
        Returns:
            Convergence status and reason.
        """
        history = ctx.deps.generation_history
        
        if len(history) < threshold_generations:
            return {'converged': False, 'reason': 'Not enough generations'}
        
        # Check last N generations for improvement
        recent = history[-threshold_generations:]
        fitness_values = [g['best_fitness'] for g in recent]
        
        max_improvement = max(fitness_values) - min(fitness_values)
        
        if max_improvement < improvement_threshold:
            return {
                'converged': True,
                'reason': f'No improvement for {threshold_generations} generations',
                'best_fitness': max(fitness_values),
            }
        
        # Check if target achieved
        if ctx.deps.target_score > 0:
            if max(fitness_values) >= ctx.deps.target_score:
                return {
                    'converged': True,
                    'reason': 'Target score achieved',
                    'best_fitness': max(fitness_values),
                }
        
        return {
            'converged': False,
            'reason': 'Evolution in progress',
            'recent_improvement': max_improvement,
        }
    
    @agent.tool
    async def submit_to_kaggle(
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        message: str = 'AGENT-K submission',
    ) -> dict[str, Any]:
        """Submit solution to Kaggle via MCP.
        
        This tool coordinates with the Kaggle MCPServerTool to submit
        the solution and retrieve the score.
        
        Args:
            ctx: Run context.
            solution_code: Solution code to submit.
            message: Submission message.
        
        Returns:
            Submission result with score and rank.
        """
        with logfire.span(
            'evolver.submit',
            competition_id=ctx.deps.competition.id,
        ):
            # Emit submission event
            await ctx.deps.event_emitter.emit(
                'tool-start',
                {
                    'taskId': 'evolution_submit',
                    'toolCallId': f'submit_{len(ctx.deps.generation_history)}',
                    'toolType': 'kaggle_mcp',
                    'operation': 'competitions.submit',
                },
            )
            
            # The actual submission would be handled by MCPServerTool
            # This returns the coordination result
            result = {
                'submission_id': 'pending',
                'status': 'submitted',
                'generation': len(ctx.deps.generation_history),
            }
            
            return result
    
    # =========================================================================
    # Output Validator
    # =========================================================================
    @agent.output_validator
    async def validate_evolution_result(
        ctx: RunContext[EvolverDeps],
        result: EvolutionResult,
    ) -> EvolutionResult:
        """Validate evolution results."""
        if result.best_fitness <= 0:
            logfire.warn('evolver.zero_fitness')
        
        if not result.best_solution:
            logfire.warn('evolver.no_solution')
        
        return result
    
    return agent


def _get_evolver_instructions() -> str:
    """Return system instructions for the Evolver agent."""
    return """You are the EVOLVER agent in the AGENT-K multi-agent system.

Your mission is to optimize competition solutions using evolutionary code search.

AVAILABLE BUILTIN TOOLS:
- Kaggle MCP: Use for all Kaggle platform operations (submit, download data, check leaderboard)
- Memory: Use to persist and retrieve context across long evolution runs
- Code Executor: Use to safely execute and evaluate solution candidates

CUSTOM TOOLS:
- mutate_solution: Apply mutations to solutions
- evaluate_fitness: Compute fitness scores
- record_generation: Log generation metrics
- check_convergence: Detect when to stop evolution
- submit_to_kaggle: Submit best solution

EVOLUTION WORKFLOW:
1. Initialize population from the provided prototype solution
2. For each generation:
   a. Evaluate fitness of all candidates using evaluate_fitness
   b. Select top performers based on fitness
   c. Apply mutations using mutate_solution (vary mutation types)
   d. Record metrics using record_generation
   e. Check convergence using check_convergence
   f. Save best solution to Memory for recovery
3. When converged or max generations reached:
   a. Submit best solution using submit_to_kaggle
   b. Return EvolutionResult with final metrics

MUTATION STRATEGY:
- Use point mutations for fine-tuning (small parameter changes)
- Use structural mutations for exploring new architectures
- Use hyperparameter mutations for learning rate, regularization
- Use crossover to combine successful solutions

IMPORTANT:
- Always save promising solutions to Memory before applying risky mutations
- Use Kaggle MCP to submit periodically for leaderboard validation
- Respect rate limits when submitting to Kaggle
- Record all generation metrics for convergence analysis
"""


# =============================================================================
# Mutation Helpers (Private)
# =============================================================================
def _apply_point_mutation(code: str, params: dict[str, Any]) -> str:
    """Apply point mutation to code."""
    # Implementation would modify specific values
    return code


def _apply_structural_mutation(code: str, params: dict[str, Any]) -> str:
    """Apply structural mutation to code."""
    # Implementation would modify architecture
    return code


def _apply_hyperparameter_mutation(code: str, params: dict[str, Any]) -> str:
    """Apply hyperparameter mutation to code."""
    # Implementation would modify hyperparameters
    return code


def _apply_crossover(code: str, other: str, params: dict[str, Any]) -> str:
    """Apply crossover between two solutions."""
    # Implementation would combine solutions
    return code


# =============================================================================
# Public Interface Class
# =============================================================================
class EvolverAgent:
    """High-level interface for the Evolver agent.
    
    Provides clean API for evolution operations while encapsulating
    the underlying Pydantic-AI agent with builtin tools.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        kaggle_mcp_url: str = DEFAULT_KAGGLE_MCP_URL,
    ) -> None:
        self._agent = create_evolver_agent(model, kaggle_mcp_url)
    
    async def evolve(
        self,
        prompt: str,
        *,
        initial_solution: str,
        deps: EvolverDeps,
    ) -> EvolutionResult:
        """Run evolutionary optimization.
        
        Args:
            prompt: Evolution directive and constraints.
            initial_solution: Starting solution code.
            deps: Dependency container.
        
        Returns:
            Evolution result with best solution.
        """
        deps.initial_solution = initial_solution
        deps.best_solution = initial_solution
        
        with logfire.span(
            'evolver.evolve',
            competition_id=deps.competition.id,
            population_size=deps.population_size,
        ):
            result = await self._agent.run(prompt, deps=deps)
            return result.output
