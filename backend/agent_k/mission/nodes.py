"""State machine nodes for AGENT-K mission graph.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import asyncio
import csv
import math
import os
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final

# Third-party (alphabetical)
import logfire
from pydantic_graph import BaseNode, End, GraphRunContext

# Local imports (core first, then alphabetical)
from ..agents import get_agent
from ..agents.evolver import EvolutionFailure, EvolverAgent, EvolverDeps, settings as evolver_settings
from ..agents.lobbyist import LobbyistDeps
from ..agents.scientist import ScientistDeps
from ..core.constants import (
    DISCOVERY_TIMEOUT_SECONDS,
    EVOLUTION_TIMEOUT_SECONDS,
    PROTOTYPE_TIMEOUT_SECONDS,
    RESEARCH_TIMEOUT_SECONDS,
    SUBMISSION_TIMEOUT_SECONDS,
)
from ..core.data import infer_competition_schema, locate_data_files, stage_competition_data
from ..core.exceptions import classify_error
from ..core.models import (
    EvaluationMetric,
    EvolutionState,
    GenerationMetrics,
    LeaderboardAnalysis,
    MissionCriteria,
    ResearchFindings,
)
from ..core.solution import execute_solution, parse_baseline_score
from .state import GraphContext, MissionResult, MissionState

if TYPE_CHECKING:
    import httpx
    from pydantic_ai import Agent

    from ..core.protocols import PlatformAdapter
    from ..ui.ag_ui import EventEmitter

__all__ = ('DiscoveryNode', 'ResearchNode', 'PrototypeNode', 'EvolutionNode', 'SubmissionNode')

_DISALLOWED_LIBRARIES: Final[tuple[str, ...]] = ('xgboost', 'lightgbm', 'catboost')


@dataclass
class DiscoveryNode(BaseNode[MissionState, GraphContext, MissionResult]):
    """Discovery phase node.

    Executes the LOBBYIST agent to discover competitions matching criteria.

    Transitions:
        - Success → ResearchNode
        - Failure → End(failure)
    """

    timeout: int = DISCOVERY_TIMEOUT_SECONDS

    async def run(self, ctx: GraphRunContext[MissionState, GraphContext]) -> ResearchNode | End[MissionResult]:
        """Execute discovery phase."""
        state = ctx.state
        emitter, http_client, platform_adapter = _require_context(ctx.deps)

        with logfire.span('graph.discovery', mission_id=state.mission_id):
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
            state.phase_started_at = datetime.now(UTC)

            try:
                if state.competition_id:
                    competition = await platform_adapter.get_competition(state.competition_id)
                    state.selected_competition = competition
                    state.discovered_competitions = [competition]
                    state.phases_completed.append('discovery')
                    await emitter.emit_phase_complete(
                        phase='discovery', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                    )
                    return ResearchNode()

                # Build prompt from criteria
                prompt = self._build_discovery_prompt(state.criteria)

                # Create dependencies
                deps = LobbyistDeps(http_client=http_client, platform_adapter=platform_adapter, event_emitter=emitter)

                # Run lobbyist agent
                lobbyist_agent = _resolve_agent(ctx.deps, 'lobbyist')
                run_result = await lobbyist_agent.run(prompt, deps=deps)
                result = run_result.output

                # Update state
                state.discovered_competitions = result.competitions

                if not result.competitions:
                    await emitter.emit_phase_complete(
                        phase='discovery', success=False, duration_ms=self._elapsed_ms(state.phase_started_at)
                    )
                    return End(
                        MissionResult(
                            success=False,
                            mission_id=state.mission_id,
                            error_message='No competitions found matching criteria',
                            phases_completed=list(state.phases_completed),
                        )
                    )

                # Select best competition
                state.selected_competition = result.competitions[0]
                state.competition_id = state.selected_competition.id
                state.phases_completed.append('discovery')

                await emitter.emit_phase_complete(
                    phase='discovery', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                )

                # Transition to research
                return ResearchNode()

            except Exception as e:
                logfire.error('discovery_failed', error=str(e), traceback=traceback.format_exc())
                state.errors.append(
                    {
                        'phase': 'discovery',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now(UTC).isoformat(),
                    }
                )
                await _emit_phase_failure(state=state, emitter=emitter, phase='discovery', error=e, context='discovery')
                return End(
                    MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message=f'Discovery failed: {e}',
                        phases_completed=list(state.phases_completed),
                    )
                )

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

        parts.append(f'- Target top {criteria.target_leaderboard_percentile * 100:.0f}% on leaderboard')

        return '\n'.join(parts)

    def _elapsed_ms(self, start: datetime | None) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.now(UTC) - start).total_seconds() * 1000) if start else 0


@dataclass
class ResearchNode(BaseNode[MissionState, GraphContext, MissionResult]):
    """Research phase node.

    Executes the SCIENTIST agent to analyze the competition.

    Transitions:
        - Success → PrototypeNode
        - Failure → End(failure)
    """

    timeout: int = RESEARCH_TIMEOUT_SECONDS

    async def run(self, ctx: GraphRunContext[MissionState, GraphContext]) -> PrototypeNode | End[MissionResult]:
        """Execute research phase."""
        state = ctx.state
        emitter, http_client, platform_adapter = _require_context(ctx.deps)
        competition = state.selected_competition
        if competition is None:
            return End(
                MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message='No competition selected for research',
                    phases_completed=list(state.phases_completed),
                )
            )

        with logfire.span('graph.research', competition_id=state.competition_id):
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
            state.phase_started_at = datetime.now(UTC)

            try:
                # Research implementation
                deps = ScientistDeps(
                    http_client=http_client, platform_adapter=platform_adapter, competition=competition
                )

                prompt = f'Research competition: {competition.title}'
                scientist_agent = _resolve_agent(ctx.deps, 'scientist')
                run_result = await scientist_agent.run(prompt, deps=deps)
                result = run_result.output

                try:
                    leaderboard = await platform_adapter.get_leaderboard(competition.id, limit=100)
                    analysis = _build_leaderboard_analysis(
                        leaderboard, state.criteria.target_leaderboard_percentile, competition.metric_direction
                    )
                except Exception as exc:
                    logfire.warning('leaderboard_analysis_failed', error=str(exc))
                    analysis = None

                state.research_findings = _build_research_findings(result, analysis)
                state.phases_completed.append('research')

                await emitter.emit_phase_complete(
                    phase='research', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                )

                return PrototypeNode()

            except Exception as e:
                logfire.error('research_failed', error=str(e), traceback=traceback.format_exc())
                state.errors.append(
                    {
                        'phase': 'research',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now(UTC).isoformat(),
                    }
                )
                await _emit_phase_failure(state=state, emitter=emitter, phase='research', error=e, context='research')
                return End(
                    MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message=f'Research failed: {e}',
                        phases_completed=list(state.phases_completed),
                    )
                )

    def _elapsed_ms(self, start: datetime | None) -> int:
        return int((datetime.now(UTC) - start).total_seconds() * 1000) if start else 0


@dataclass
class PrototypeNode(BaseNode[MissionState, GraphContext, MissionResult]):
    """Prototype phase node.

    Generates initial baseline solution.

    Transitions:
        - Success → EvolutionNode
        - Failure → End(failure)
    """

    timeout: int = PROTOTYPE_TIMEOUT_SECONDS

    async def run(self, ctx: GraphRunContext[MissionState, GraphContext]) -> EvolutionNode | End[MissionResult]:
        """Execute prototype phase."""
        state = ctx.state
        emitter, _http_client, platform_adapter = _require_context(ctx.deps)
        competition = state.selected_competition
        if competition is None:
            return End(
                MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message='No competition selected for prototype phase',
                    phases_completed=list(state.phases_completed),
                )
            )

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
            state.phase_started_at = datetime.now(UTC)

            try:
                with tempfile.TemporaryDirectory() as work_dir:
                    work_path = Path(work_dir)
                    competition_id = state.competition_id or competition.id
                    state.competition_id = competition_id
                    data_files = await platform_adapter.download_data(competition_id, work_dir)
                    train_path, test_path, sample_path = locate_data_files(data_files)
                    staged = stage_competition_data(
                        train_path, test_path, sample_path, work_path, competition_id=competition_id
                    )
                    schema = infer_competition_schema(staged['train'], staged['test'], staged['sample'])

                    prototype_code = self._generate_prototype(
                        competition,
                        state.research_findings,
                        target_columns=schema.target_columns,
                        train_target_columns=schema.train_target_columns,
                        id_column=schema.id_column,
                    )

                    execution = await execute_solution(
                        prototype_code,
                        work_path,
                        timeout_seconds=self.timeout,
                        use_builtin_code_execution=True,
                        model_spec=evolver_settings.model,
                    )

                    submission_path = work_path / 'submission.csv'
                    baseline_score = parse_baseline_score(execution.stdout)
                    if not submission_path.exists() or execution.returncode != 0 or execution.timed_out:
                        fallback_code = _generate_fallback_prototype(
                            target_columns=schema.target_columns,
                            train_target_columns=schema.train_target_columns,
                            id_column=schema.id_column,
                            metric=competition.metric,
                        )
                        _write_fallback_submission(
                            train_path=staged['train'],
                            test_path=staged['test'],
                            sample_path=staged['sample'],
                            metric=competition.metric,
                            output_path=submission_path,
                        )
                        prototype_code = fallback_code

                    if baseline_score is None:
                        baseline_score = _compute_baseline_score(
                            train_path=staged['train'],
                            target_columns=schema.train_target_columns,
                            metric=competition.metric,
                        )

                    state.prototype_code = prototype_code
                    state.prototype_score = baseline_score

                state.phases_completed.append('prototype')

                await emitter.emit_phase_complete(
                    phase='prototype', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                )

                return EvolutionNode()

            except Exception as e:
                logfire.error('prototype_failed', error=str(e), traceback=traceback.format_exc())
                state.errors.append(
                    {
                        'phase': 'prototype',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now(UTC).isoformat(),
                    }
                )
                await _emit_phase_failure(state=state, emitter=emitter, phase='prototype', error=e, context='prototype')
                return End(
                    MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message=f'Prototype failed: {e}',
                        phases_completed=list(state.phases_completed),
                    )
                )

    def _generate_prototype(
        self,
        competition: Any,
        research: Any,
        *,
        target_columns: list[str],
        train_target_columns: list[str],
        id_column: str,
    ) -> str:
        """Generate prototype solution code."""
        metric = getattr(competition, 'metric', None)
        metric_key = metric if isinstance(metric, EvaluationMetric) else EvaluationMetric.ACCURACY
        metric_value = metric_key.value
        target_columns_repr = repr(target_columns)
        train_target_columns_repr = repr(train_target_columns)
        strategy_items: list[str] = []
        for attr in ('strategy_recommendations', 'recommended_approaches'):
            value = getattr(research, attr, None)
            if isinstance(value, list):
                strategy_items.extend(str(item) for item in value if item)
            elif isinstance(value, str) and value:
                strategy_items.append(value)
        strategy_text = ' '.join(strategy_items)
        strategy_lower = strategy_text.lower()

        is_classification = metric_key in {
            EvaluationMetric.ACCURACY,
            EvaluationMetric.AUC,
            EvaluationMetric.LOG_LOSS,
            EvaluationMetric.F1,
        }
        uses_proba = metric_key in {EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS}

        if 'linear' in strategy_lower:
            model_class = 'LogisticRegression' if is_classification else 'LinearRegression'
            model_import = 'from sklearn.linear_model import LogisticRegression, LinearRegression'
        elif 'gradient' in strategy_lower or 'boost' in strategy_lower:
            model_class = 'GradientBoostingClassifier' if is_classification else 'GradientBoostingRegressor'
            model_import = 'from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor'
        else:
            model_class = 'RandomForestClassifier' if is_classification else 'RandomForestRegressor'
            model_import = 'from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor'

        return (
            dedent(
                f"""
        import os
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            mean_squared_log_error,
            roc_auc_score,
        )
        {model_import}
        
        TARGET_COLUMNS = {target_columns_repr}
        TRAIN_TARGET_COLUMNS = {train_target_columns_repr}
        ID_COLUMN = "{id_column}"
        METRIC = "{metric_value}"
        METRIC_KEY = METRIC.lower().replace("_", "")
        VALIDATION_SPLIT = float(os.getenv("AGENT_K_VALIDATION_SPLIT", "0.2"))
        IS_CLASSIFICATION = {is_classification}
        USES_PROBA = {uses_proba}
        
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        
        y = train_df[TRAIN_TARGET_COLUMNS]
        X = train_df.drop(columns=TRAIN_TARGET_COLUMNS)
        
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, categorical_cols),
                ("numeric", numeric_transformer, numeric_cols),
            ],
        )
        
        base_model = {model_class}(random_state=42)
        if len(TRAIN_TARGET_COLUMNS) > 1:
            base_model = (
                MultiOutputClassifier(base_model)
                if IS_CLASSIFICATION
                else MultiOutputRegressor(base_model)
            )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SPLIT,
            random_state=42,
            stratify=y if IS_CLASSIFICATION and len(TRAIN_TARGET_COLUMNS) == 1 else None,
        )
        
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ])
        
        def _encode_labels(values, classes):
            mapping = {{label: idx for idx, label in enumerate(classes)}}
            return np.array([mapping.get(value, -1) for value in values])
        
        def _score_classification_single(y_true, preds, probas, classes):
            if METRIC_KEY == "accuracy":
                return accuracy_score(y_true, preds)
            if METRIC_KEY == "f1":
                average = "binary" if len(classes) == 2 else "weighted"
                return f1_score(y_true, preds, average=average, zero_division=0)
            if METRIC_KEY == "auc":
                if len(classes) < 2:
                    return 0.5
                if len(classes) == 2:
                    pos_label = classes[1]
                    y_binary = (y_true == pos_label).astype(int)
                    return roc_auc_score(y_binary, probas[:, 1])
                y_encoded = _encode_labels(y_true, classes)
                return roc_auc_score(
                    y_encoded,
                    probas,
                    multi_class="ovo",
                    average="macro",
                )
            if METRIC_KEY == "logloss":
                if len(classes) < 2:
                    return 0.0
                return log_loss(y_true, probas, labels=classes)
            return accuracy_score(y_true, preds)
        
        def _score_classification(y_true, preds, probas, model_step):
            scores = []
            if len(TRAIN_TARGET_COLUMNS) == 1:
                classes = model_step.classes_
                scores.append(_score_classification_single(y_true, preds, probas, classes))
            else:
                for idx, column in enumerate(TRAIN_TARGET_COLUMNS):
                    estimator = model_step.estimators_[idx]
                    classes = estimator.classes_
                    col_preds = preds[:, idx]
                    col_probas = probas[idx] if probas is not None else None
                    scores.append(
                        _score_classification_single(
                            y_true[column],
                            col_preds,
                            col_probas,
                            classes,
                        )
                    )
            return float(np.mean(scores)) if scores else 0.0
        
        def _score_regression_single(y_true, preds):
            if METRIC_KEY == "rmse":
                return mean_squared_error(y_true, preds, squared=False)
            if METRIC_KEY == "mae":
                return mean_absolute_error(y_true, preds)
            if METRIC_KEY == "rmsle":
                return mean_squared_log_error(y_true, preds) ** 0.5
            return mean_squared_error(y_true, preds, squared=False)
        
        def _score_regression(y_true, preds):
            if len(TRAIN_TARGET_COLUMNS) == 1:
                return _score_regression_single(y_true, preds)
            scores = [
                _score_regression_single(y_true[column], preds[:, idx])
                for idx, column in enumerate(TRAIN_TARGET_COLUMNS)
            ]
            return float(np.mean(scores)) if scores else 0.0
        
        clf.fit(X_train, y_train)
        model_step = clf.named_steps["model"]
        if IS_CLASSIFICATION:
            if USES_PROBA:
                val_probas = clf.predict_proba(X_val)
                val_preds = clf.predict(X_val)
                score = _score_classification(y_val, val_preds, val_probas, model_step)
            else:
                val_preds = clf.predict(X_val)
                score = _score_classification(y_val, val_preds, None, model_step)
        else:
            val_preds = clf.predict(X_val)
            score = _score_regression(y_val, val_preds)
        print(f"Baseline {metric_value} score: {{score}}")
        
        clf.fit(X, y)
        submission = pd.read_csv("sample_submission.csv")
        if IS_CLASSIFICATION and USES_PROBA:
            if len(TRAIN_TARGET_COLUMNS) == 1 and len(TARGET_COLUMNS) > 1:
                test_probas = clf.predict_proba(test_df)
                class_labels = [str(label) for label in model_step.classes_]
                proba_df = pd.DataFrame(test_probas, columns=class_labels)

                def _normalize_label(label):
                    return str(label).lower().replace("class_", "").replace("class ", "")

                normalized = {{_normalize_label(label): label for label in class_labels}}
                for column in TARGET_COLUMNS:
                    key = _normalize_label(column)
                    label = normalized.get(key)
                    if label and label in proba_df.columns:
                        submission[column] = proba_df[label]
                    elif test_probas.shape[1] == len(TARGET_COLUMNS):
                        submission[column] = test_probas[:, TARGET_COLUMNS.index(column)]
                    else:
                        submission[column] = 0.0
            elif len(TRAIN_TARGET_COLUMNS) == 1:
                test_probas = clf.predict_proba(test_df)
                submission[TARGET_COLUMNS[0]] = test_probas[:, 1]
            else:
                test_probas = clf.predict_proba(test_df)
                for idx, column in enumerate(TARGET_COLUMNS):
                    column_proba = test_probas[idx]
                    if column_proba.ndim == 2 and column_proba.shape[1] > 1:
                        submission[column] = column_proba[:, 1]
                    else:
                        submission[column] = column_proba.ravel()
        else:
            test_preds = clf.predict(test_df)
            if len(TARGET_COLUMNS) == 1:
                submission[TARGET_COLUMNS[0]] = test_preds
            else:
                for idx, column in enumerate(TARGET_COLUMNS):
                    submission[column] = test_preds[:, idx]
        submission.to_csv("submission.csv", index=False)
        """
            ).strip()
            + '\n'
        )

    def _elapsed_ms(self, start: datetime | None) -> int:
        return int((datetime.now(UTC) - start).total_seconds() * 1000) if start else 0


@dataclass
class EvolutionNode(BaseNode[MissionState, GraphContext, MissionResult]):
    """Evolution phase node.

    Executes the EVOLVER agent to optimize the solution.

    Transitions:
        - Success → SubmissionNode
        - Failure → End(failure with best solution)
    """

    timeout: int = EVOLUTION_TIMEOUT_SECONDS

    async def run(self, ctx: GraphRunContext[MissionState, GraphContext]) -> SubmissionNode | End[MissionResult]:  # noqa: C901
        """Execute evolution phase."""
        state = ctx.state
        emitter, _http_client, platform_adapter = _require_context(ctx.deps)
        competition = state.selected_competition
        if competition is None:
            return End(
                MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message='No competition selected for evolution phase',
                    phases_completed=list(state.phases_completed),
                )
            )

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
            state.phase_started_at = datetime.now(UTC)

            try:
                # Initialize evolution state
                state.evolution_state = EvolutionState(
                    max_generations=state.criteria.max_evolution_rounds,
                    min_improvements_required=state.criteria.min_improvements_required,
                )

                if _quick_test_enabled(state.criteria):
                    return await self._run_quick_evolution(state, emitter)

                with tempfile.TemporaryDirectory() as work_dir:
                    work_path = Path(work_dir)
                    competition_id = state.competition_id or competition.id
                    state.competition_id = competition_id
                    data_files = await platform_adapter.download_data(competition_id, work_dir)
                    train_path, test_path, sample_path = locate_data_files(data_files)
                    staged = stage_competition_data(
                        train_path, test_path, sample_path, work_path, competition_id=competition_id
                    )
                    schema = infer_competition_schema(staged['train'], staged['test'], staged['sample'])

                    population_size = evolver_settings.population_size
                    solution_timeout = evolver_settings.solution_timeout
                    if state.criteria.max_evolution_rounds <= 5:
                        population_size = min(population_size, 4)
                        solution_timeout = min(solution_timeout, 90)
                    elif state.criteria.max_evolution_rounds <= 10:
                        population_size = min(population_size, 10)
                        solution_timeout = min(solution_timeout, 180)

                    min_generations = min(evolver_settings.min_generations, state.criteria.max_evolution_rounds)
                    min_improvements_required = state.criteria.min_improvements_required
                    target_score = self._calculate_target_score(state)
                    evolution_models = [model.strip() for model in state.criteria.evolution_models if model.strip()]
                    strategy_recommendations = (
                        state.research_findings.strategy_recommendations if state.research_findings else []
                    )
                    filtered_recommendations = _filter_disallowed_recommendations(strategy_recommendations)
                    strategy_text = '; '.join(filtered_recommendations) if filtered_recommendations else 'N/A'
                    base_prompt = f"""
                    Evolve solution for {competition.title}.
                    Target: Top {state.criteria.target_leaderboard_percentile * 100:.0f}% on leaderboard.
                    Research suggests: {strategy_text}
                    Minimum generations before convergence (overall): {min_generations}. Do not treat this as a per-segment requirement.
                    Minimum improvements required before submission: {min_improvements_required}.
                    Maintain diversity using model families and solution complexity bins.
                    Use sample_elites to pull top and diverse candidates.
                    Use cascade evaluation in evaluate_fitness to skip full runs when quick checks fail.
                    Use only numpy, pandas, and scikit-learn. Avoid xgboost, lightgbm, and catboost.
                    If research mentions disallowed libraries, ignore those suggestions and stay within the allowed stack.
                    """

                    baseline_fitness = _fitness_from_score(state.prototype_score, competition.metric_direction)
                    best_solution = state.prototype_code or ''
                    best_fitness: float | None = baseline_fitness
                    improvement_count = 0
                    combined_history: list[dict[str, Any]] = []
                    elite_archive: dict[tuple[int, str], Any] = {}
                    convergence_detected = False
                    convergence_reason: str | None = None

                    def record_rate_limit(
                        error_message: str, *, model_spec: str | None, error_type: str | None
                    ) -> None:
                        state.errors.append(
                            {
                                'phase': 'evolution',
                                'error': error_message,
                                'error_type': error_type or 'rate_limit',
                                'timestamp': datetime.now(UTC).isoformat(),
                            }
                        )
                        logfire.warning(
                            'evolution_rate_limited', model=model_spec or evolver_settings.model, error=error_message
                        )

                    deps_kwargs = {
                        'competition': competition,
                        'event_emitter': emitter,
                        'platform_adapter': platform_adapter,
                        'data_dir': work_path,
                        'train_path': staged['train'],
                        'test_path': staged['test'],
                        'sample_path': staged['sample'],
                        'target_columns': schema.target_columns,
                        'train_target_columns': schema.train_target_columns,
                        'population_size': population_size,
                        'solution_timeout': solution_timeout,
                        'target_score': target_score,
                        'min_generations': min_generations,
                        'min_improvements_required': min_improvements_required,
                    }

                    if not evolution_models:
                        deps = EvolverDeps(
                            **deps_kwargs,  # type: ignore[arg-type]
                            initial_solution=best_solution,
                            best_solution=best_solution or None,
                            best_fitness=best_fitness,
                            max_generations=state.criteria.max_evolution_rounds,
                            improvement_count=improvement_count,
                            elite_archive=elite_archive,
                        )
                        evolver_agent = _resolve_agent(ctx.deps, 'evolver')
                        try:
                            run_result = await evolver_agent.run(base_prompt, deps=deps)
                        except Exception as exc:
                            if _is_rate_limit_error(exc):
                                record_rate_limit(str(exc), model_spec=None, error_type=getattr(exc, 'code', None))
                                combined_history = deps.generation_history
                                improvement_count = deps.improvement_count
                                convergence_detected = True
                                convergence_reason = 'rate_limit'
                            else:
                                raise
                        else:
                            result = run_result.output
                            if isinstance(result, EvolutionFailure):
                                is_rate_limited = _is_rate_limit_error(result.error_message) or _is_rate_limit_error(
                                    result.error_type
                                )
                                if is_rate_limited:
                                    if result.partial_solution:
                                        best_solution = result.partial_solution
                                    record_rate_limit(
                                        result.error_message, model_spec=None, error_type=result.error_type
                                    )
                                    combined_history = deps.generation_history
                                    improvement_count = deps.improvement_count
                                    convergence_detected = True
                                    convergence_reason = 'rate_limit'
                                elif _is_constraints_failure(result.error_message):
                                    best_solution = result.partial_solution or state.prototype_code or ''
                                    best_fitness = baseline_fitness or 0.0
                                    combined_history = deps.generation_history
                                    improvement_count = deps.improvement_count
                                    convergence_detected = True
                                    convergence_reason = 'constraints'
                                    logfire.warning('evolution_constraints_fallback', error=result.error_message)
                                    state.errors.append(
                                        {
                                            'phase': 'evolution',
                                            'error': result.error_message,
                                            'error_type': result.error_type or 'constraints',
                                            'timestamp': datetime.now(UTC).isoformat(),
                                        }
                                    )
                                    await emitter.emit_error(
                                        error_id=f'evolution_constraints_{state.mission_id}',
                                        category='recoverable',
                                        error_type=result.error_type or 'constraints',
                                        message=result.error_message,
                                        context='evolution',
                                        recovery_strategy='fallback',
                                    )
                                else:
                                    if result.partial_solution and state.evolution_state is not None:
                                        state.evolution_state = state.evolution_state.model_copy(
                                            update={'best_solution': {'code': result.partial_solution, 'fitness': 0.0}}
                                        )

                                    state.errors.append(
                                        {
                                            'phase': 'evolution',
                                            'error': result.error_message,
                                            'error_type': result.error_type,
                                            'timestamp': datetime.now(UTC).isoformat(),
                                        }
                                    )

                                    await emitter.emit_phase_error(
                                        phase='evolution',
                                        error=result.error_message,
                                        recoverable=bool(result.recoverable),
                                    )
                                    await emitter.emit_error(
                                        error_id=f'evolution_{state.mission_id}',
                                        category='recoverable' if result.recoverable else 'fatal',
                                        error_type=result.error_type,
                                        message=result.error_message,
                                        context='evolution',
                                        recovery_strategy='retry' if result.recoverable else 'abort',
                                    )
                                    await emitter.emit_phase_complete(
                                        phase='evolution',
                                        success=False,
                                        duration_ms=self._elapsed_ms(state.phase_started_at),
                                    )

                                    return End(
                                        MissionResult(
                                            success=False,
                                            mission_id=state.mission_id,
                                            competition_id=state.competition_id,
                                            error_message=f'Evolution failed: {result.error_message}',
                                            phases_completed=list(state.phases_completed),
                                        )
                                    )

                            if not isinstance(result, EvolutionFailure):
                                combined_history = deps.generation_history
                                improvement_count = deps.improvement_count
                                best_solution = result.best_solution
                                best_fitness = result.best_fitness
                                convergence_detected = result.convergence_achieved
                                convergence_reason = result.convergence_reason
                    else:
                        agents_by_model: dict[str, Any] = {}
                        remaining_generations = state.criteria.max_evolution_rounds
                        available_models = [model for model in evolution_models if model]
                        segment_index = 0
                        model_index = 0

                        while remaining_generations > 0 and available_models:
                            segment_index += 1
                            rotation_stride = max(
                                5, min(25, math.ceil(remaining_generations / max(len(available_models), 1)))
                            )
                            model_spec = available_models[model_index % len(available_models)]
                            segment_generations = min(rotation_stride, remaining_generations)
                            generation_offset = len(combined_history)

                            agent = agents_by_model.get(model_spec)
                            if agent is None:
                                segment_settings = evolver_settings.model_copy(update={'model': model_spec})
                                agent_instance = EvolverAgent(settings=segment_settings, register=False)
                                agent = agent_instance.agent
                                agents_by_model[model_spec] = agent

                            segment_prompt = (
                                f"""{base_prompt}
Model rotation segment {segment_index} using {model_spec}."""
                                f'\nRun {segment_generations} generations for this segment starting at {generation_offset + 1}.'
                                f' These segments roll up to the overall target of {state.criteria.max_evolution_rounds} generations.'
                            )

                            segment_deps = EvolverDeps(
                                **deps_kwargs,  # type: ignore[arg-type]
                                initial_solution=best_solution or state.prototype_code or '',
                                best_solution=best_solution or None,
                                best_fitness=best_fitness,
                                max_generations=segment_generations,
                                improvement_count=improvement_count,
                                generation_history=combined_history,
                                generation_offset=generation_offset,
                                elite_archive=elite_archive,
                            )
                            try:
                                run_result = await agent.run(segment_prompt, deps=segment_deps)
                            except Exception as exc:
                                if _is_rate_limit_error(exc):
                                    record_rate_limit(
                                        str(exc), model_spec=model_spec, error_type=getattr(exc, 'code', None)
                                    )
                                    agents_by_model.pop(model_spec, None)
                                    available_models = [model for model in available_models if model != model_spec]
                                    if not available_models:
                                        convergence_detected = True
                                        convergence_reason = 'rate_limit'
                                    continue
                                raise

                            result = run_result.output
                            if isinstance(result, EvolutionFailure):
                                is_rate_limited = _is_rate_limit_error(result.error_message) or _is_rate_limit_error(
                                    result.error_type
                                )
                                if is_rate_limited:
                                    if result.partial_solution:
                                        best_solution = result.partial_solution
                                    record_rate_limit(
                                        result.error_message, model_spec=model_spec, error_type=result.error_type
                                    )
                                    improvement_count = segment_deps.improvement_count
                                    agents_by_model.pop(model_spec, None)
                                    available_models = [model for model in available_models if model != model_spec]
                                    if not available_models:
                                        convergence_detected = True
                                        convergence_reason = 'rate_limit'
                                    continue
                                if _is_constraints_failure(result.error_message):
                                    if result.partial_solution:
                                        best_solution = result.partial_solution
                                    elif not best_solution:
                                        best_solution = state.prototype_code or ''
                                    if best_fitness is None:
                                        best_fitness = baseline_fitness or 0.0
                                    combined_history = segment_deps.generation_history
                                    improvement_count = segment_deps.improvement_count
                                    convergence_detected = True
                                    convergence_reason = 'constraints'
                                    logfire.warning(
                                        'evolution_constraints_fallback', error=result.error_message, model=model_spec
                                    )
                                    state.errors.append(
                                        {
                                            'phase': 'evolution',
                                            'error': result.error_message,
                                            'error_type': result.error_type or 'constraints',
                                            'timestamp': datetime.now(UTC).isoformat(),
                                        }
                                    )
                                    await emitter.emit_error(
                                        error_id=f'evolution_constraints_{state.mission_id}',
                                        category='recoverable',
                                        error_type=result.error_type or 'constraints',
                                        message=result.error_message,
                                        context='evolution',
                                        recovery_strategy='fallback',
                                    )
                                    break

                                if result.partial_solution and state.evolution_state is not None:
                                    state.evolution_state = state.evolution_state.model_copy(
                                        update={'best_solution': {'code': result.partial_solution, 'fitness': 0.0}}
                                    )

                                state.errors.append(
                                    {
                                        'phase': 'evolution',
                                        'error': result.error_message,
                                        'error_type': result.error_type,
                                        'timestamp': datetime.now(UTC).isoformat(),
                                    }
                                )

                                await emitter.emit_phase_error(
                                    phase='evolution', error=result.error_message, recoverable=bool(result.recoverable)
                                )
                                await emitter.emit_error(
                                    error_id=f'evolution_{state.mission_id}',
                                    category='recoverable' if result.recoverable else 'fatal',
                                    error_type=result.error_type,
                                    message=result.error_message,
                                    context='evolution',
                                    recovery_strategy='retry' if result.recoverable else 'abort',
                                )
                                await emitter.emit_phase_complete(
                                    phase='evolution',
                                    success=False,
                                    duration_ms=self._elapsed_ms(state.phase_started_at),
                                )

                                return End(
                                    MissionResult(
                                        success=False,
                                        mission_id=state.mission_id,
                                        competition_id=state.competition_id,
                                        error_message=f'Evolution failed: {result.error_message}',
                                        phases_completed=list(state.phases_completed),
                                    )
                                )

                            if best_fitness is None or result.best_fitness > best_fitness:
                                best_fitness = result.best_fitness
                                best_solution = result.best_solution
                            improvement_count = segment_deps.improvement_count
                            model_index += 1

                            if result.convergence_achieved and len(combined_history) >= min_generations:
                                convergence_detected = True
                                convergence_reason = result.convergence_reason
                                break

                            if len(combined_history) <= generation_offset:
                                baseline = (
                                    result.best_fitness if result.best_fitness is not None else (best_fitness or 0.0)
                                )
                                for idx in range(segment_generations):
                                    combined_history.append(
                                        {
                                            'generation': generation_offset + idx + 1,
                                            'best_fitness': baseline,
                                            'mean_fitness': baseline,
                                            'worst_fitness': baseline,
                                            'population_size': population_size,
                                            'mutations': {
                                                'point': 0,
                                                'structural': 0,
                                                'hyperparameter': 0,
                                                'crossover': 0,
                                            },
                                        }
                                    )

                            remaining_generations = state.criteria.max_evolution_rounds - len(combined_history)

                        if not available_models and remaining_generations > 0:
                            convergence_detected = True
                            convergence_reason = 'rate_limit'

                resolved_solution = best_solution or state.prototype_code or ''
                resolved_fitness = best_fitness if best_fitness is not None else 0.0
                history = combined_history

                # Update state with evolution results
                state.evolution_state = state.evolution_state.model_copy(
                    update={
                        'best_solution': {'code': resolved_solution, 'fitness': resolved_fitness},
                        'convergence_detected': convergence_detected,
                        'convergence_reason': convergence_reason,
                        'current_generation': len(history),
                        'max_generations': state.criteria.max_evolution_rounds,
                        'population_size': population_size,
                        'improvement_count': improvement_count,
                        'min_improvements_required': min_improvements_required,
                        'generation_history': _convert_generation_history(history, population_size),
                    }
                )

                state.phases_completed.append('evolution')

                await emitter.emit_phase_complete(
                    phase='evolution', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                )

                return SubmissionNode()

            except Exception as e:
                logfire.error('evolution_failed', error=str(e), traceback=traceback.format_exc())
                state.errors.append(
                    {
                        'phase': 'evolution',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now(UTC).isoformat(),
                    }
                )
                await _emit_phase_failure(state=state, emitter=emitter, phase='evolution', error=e, context='evolution')
                # Even on failure, try to submit best solution if available
                if state.evolution_state and state.evolution_state.best_solution:
                    return SubmissionNode()
                return End(
                    MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        error_message=f'Evolution failed: {e}',
                        phases_completed=list(state.phases_completed),
                    )
                )

    async def _run_quick_evolution(self, state: MissionState, emitter: EventEmitter) -> SubmissionNode:
        max_generations = state.criteria.max_evolution_rounds
        population_size = max(2, min(evolver_settings.population_size, 2))
        baseline_score = state.prototype_score or 0.0
        mutations = {'point': 0, 'structural': 0, 'hyperparameter': 0, 'crossover': 0}

        generation_history: list[GenerationMetrics] = []
        for generation in range(1, max_generations + 1):
            await emitter.emit_generation_start(generation=generation, population_size=population_size)
            metrics = GenerationMetrics(
                generation=generation,
                best_fitness=baseline_score,
                mean_fitness=baseline_score,
                worst_fitness=baseline_score,
                population_size=population_size,
                mutations=mutations,
            )
            generation_history.append(metrics)
            await emitter.emit_generation_complete(
                generation=generation,
                best_fitness=baseline_score,
                mean_fitness=baseline_score,
                worst_fitness=baseline_score,
                population_size=population_size,
                mutations=mutations,
            )

        if state.evolution_state is not None:
            state.evolution_state = state.evolution_state.model_copy(
                update={
                    'current_generation': max_generations,
                    'max_generations': max_generations,
                    'population_size': population_size,
                    'improvement_count': 0,
                    'min_improvements_required': state.criteria.min_improvements_required,
                    'best_solution': {'code': state.prototype_code or '', 'fitness': baseline_score},
                    'generation_history': generation_history,
                    'convergence_detected': True,
                    'convergence_reason': 'quick_test_mode',
                }
            )
        state.phases_completed.append('evolution')
        await emitter.emit_phase_complete(
            phase='evolution', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
        )
        return SubmissionNode()

    def _calculate_target_score(self, state: MissionState) -> float:
        """Calculate target score from research findings."""
        override = os.getenv('AGENT_K_TARGET_SCORE')
        if override:
            try:
                return float(override)
            except ValueError:
                logfire.warning('invalid_target_score_override', value=override)
        if state.research_findings and state.research_findings.leaderboard_analysis:
            return state.research_findings.leaderboard_analysis.target_score
        return 0.0

    def _elapsed_ms(self, start: datetime | None) -> int:
        return int((datetime.now(UTC) - start).total_seconds() * 1000) if start else 0


@dataclass
class SubmissionNode(BaseNode[MissionState, GraphContext, MissionResult]):
    """Submission phase node.

    Final submission of best solution.

    Transitions:
        - Success → End(success)
        - Failure → End(failure)
    """

    timeout: int = SUBMISSION_TIMEOUT_SECONDS

    async def run(self, ctx: GraphRunContext[MissionState, GraphContext]) -> End[MissionResult]:
        """Execute submission phase."""
        state = ctx.state
        emitter, _http_client, platform_adapter = _require_context(ctx.deps)
        competition = state.selected_competition
        if competition is None:
            return End(
                MissionResult(
                    success=False,
                    mission_id=state.mission_id,
                    error_message='No competition selected for submission',
                    phases_completed=list(state.phases_completed),
                )
            )

        with logfire.span('graph.submission', competition_id=state.competition_id):
            await emitter.emit_phase_start(
                phase='submission',
                objectives=['Generate final predictions', 'Submit to Kaggle', 'Retrieve final score and rank'],
            )

            state.current_phase = 'submission'
            state.phase_started_at = datetime.now(UTC)

            try:
                min_improvements_required = (
                    state.evolution_state.min_improvements_required if state.evolution_state else 0
                )
                improvement_count = state.evolution_state.improvement_count if state.evolution_state else 0
                if min_improvements_required > 0 and improvement_count < min_improvements_required:
                    error_message = (
                        f'Minimum improvements not reached ({improvement_count}/{min_improvements_required})'
                    )
                    logfire.warning(
                        'submission_blocked_min_improvements',
                        improvement_count=improvement_count,
                        min_improvements_required=min_improvements_required,
                    )
                    state.errors.append(
                        {
                            'phase': 'submission',
                            'error': error_message,
                            'error_type': 'min_improvements',
                            'timestamp': datetime.now(UTC).isoformat(),
                        }
                    )
                    await emitter.emit_phase_error(phase='submission', error=error_message, recoverable=True)
                    await emitter.emit_phase_complete(
                        phase='submission', success=False, duration_ms=self._elapsed_ms(state.phase_started_at)
                    )
                    return End(
                        MissionResult(
                            success=False,
                            mission_id=state.mission_id,
                            competition_id=state.competition_id,
                            error_message=error_message,
                            phases_completed=list(state.phases_completed),
                        )
                    )

                # Get best solution
                best_code = ''
                if state.evolution_state and state.evolution_state.best_solution:
                    best_code = state.evolution_state.best_solution.get('code', '')
                elif state.prototype_code:
                    best_code = state.prototype_code

                if not best_code:
                    return End(
                        MissionResult(
                            success=False,
                            mission_id=state.mission_id,
                            error_message='No solution available for submission',
                            phases_completed=list(state.phases_completed),
                        )
                    )

                with tempfile.TemporaryDirectory() as work_dir:
                    work_path = Path(work_dir)
                    competition_id = state.competition_id or competition.id
                    state.competition_id = competition_id
                    data_files = await platform_adapter.download_data(competition_id, work_dir)
                    train_path, test_path, sample_path = locate_data_files(data_files)
                    staged = stage_competition_data(
                        train_path, test_path, sample_path, work_path, competition_id=competition_id
                    )

                    submission_path = work_path / 'submission.csv'
                    execution = await execute_solution(
                        best_code,
                        work_path,
                        timeout_seconds=self.timeout,
                        use_builtin_code_execution=True,
                        model_spec=evolver_settings.model,
                    )

                    if not submission_path.exists() or execution.returncode != 0 or execution.timed_out:
                        fallback_code = state.prototype_code
                        if fallback_code and fallback_code != best_code:
                            execution = await execute_solution(
                                fallback_code,
                                work_path,
                                timeout_seconds=self.timeout,
                                use_builtin_code_execution=True,
                                model_spec=evolver_settings.model,
                            )

                        if not submission_path.exists() or execution.returncode != 0 or execution.timed_out:
                            _write_fallback_submission(
                                train_path=staged['train'],
                                test_path=staged['test'],
                                sample_path=staged['sample'],
                                metric=competition.metric,
                                output_path=submission_path,
                            )

                    if not submission_path.exists():
                        return End(
                            MissionResult(
                                success=False,
                                mission_id=state.mission_id,
                                error_message='Failed to generate submission file',
                                phases_completed=list(state.phases_completed),
                            )
                        )

                    # Submit via platform adapter
                    submission = await platform_adapter.submit(
                        competition_id, str(submission_path), message=f'AGENT-K mission {state.mission_id}'
                    )

                state.final_submission_id = submission.id

                # Wait for score
                for _ in range(10):  # Poll for score
                    await asyncio.sleep(5)
                    status = await platform_adapter.get_submission_status(competition_id, submission.id)
                    if status.public_score is not None:
                        state.final_score = status.public_score
                        break

                # Get rank
                leaderboard = await platform_adapter.get_leaderboard(competition_id, limit=10000)
                for entry in leaderboard:
                    if entry.score == state.final_score:
                        state.final_rank = entry.rank
                        break

                state.phases_completed.append('submission')

                # Emit submission result
                await emitter.emit_submission_result(
                    submission_id=submission.id,
                    generation=(len(state.evolution_state.generation_history) if state.evolution_state else 0),
                    cv_score=(
                        state.evolution_state.best_solution.get('fitness', 0)
                        if state.evolution_state and state.evolution_state.best_solution
                        else 0
                    ),
                    public_score=state.final_score,
                    rank=state.final_rank,
                    total_teams=len(leaderboard),
                )

                await emitter.emit_phase_complete(
                    phase='submission', success=True, duration_ms=self._elapsed_ms(state.phase_started_at)
                )

                # Calculate total duration
                total_duration_ms = int((datetime.now(UTC) - state.started_at).total_seconds() * 1000)

                return End(
                    MissionResult(
                        success=True,
                        mission_id=state.mission_id,
                        competition_id=state.competition_id,
                        final_rank=state.final_rank,
                        final_score=state.final_score,
                        total_submissions=(
                            len(state.evolution_state.leaderboard_submissions) if state.evolution_state else 1
                        ),
                        evolution_generations=(
                            len(state.evolution_state.generation_history) if state.evolution_state else 0
                        ),
                        duration_ms=total_duration_ms,
                        phases_completed=list(state.phases_completed),
                    )
                )

            except Exception as e:
                logfire.error('submission_failed', error=str(e))
                state.errors.append(
                    {'phase': 'submission', 'error': str(e), 'timestamp': datetime.now(UTC).isoformat()}
                )
                return End(
                    MissionResult(
                        success=False,
                        mission_id=state.mission_id,
                        competition_id=state.competition_id,
                        error_message=f'Submission failed: {e}',
                        phases_completed=list(state.phases_completed),
                    )
                )

    def _elapsed_ms(self, start: datetime | None) -> int:
        return int((datetime.now(UTC) - start).total_seconds() * 1000) if start else 0


def _require_context(context: GraphContext) -> tuple[EventEmitter, httpx.AsyncClient, PlatformAdapter]:
    if context.event_emitter is None:
        raise RuntimeError('GraphContext.event_emitter is required')
    if context.http_client is None:
        raise RuntimeError('GraphContext.http_client is required')
    if context.platform_adapter is None:
        raise RuntimeError('GraphContext.platform_adapter is required')
    return context.event_emitter, context.http_client, context.platform_adapter


def _resolve_agent(context: GraphContext, name: str) -> Agent[Any, Any]:
    agents = context.agents
    if agents and name in agents:
        return agents[name]
    return get_agent(name)


def _quick_test_enabled(criteria: MissionCriteria) -> bool:
    if criteria.max_evolution_rounds > 5:
        return False
    flag = os.getenv('AGENT_K_QUICK_TEST', 'false').strip().lower()
    return flag in {'1', 'true', 'yes', 'on'}


def _is_rate_limit_error(error: Exception | str | None) -> bool:
    if not error:
        return False
    if isinstance(error, Exception):
        status_code = getattr(error, 'status_code', None)
        response_status = getattr(getattr(error, 'response', None), 'status_code', None)
        if status_code == 429 or response_status == 429:
            return True
        error_code = getattr(error, 'code', None) or getattr(error, 'error_code', None)
        if isinstance(error_code, str) and 'rate' in error_code.lower():
            return True
    message = str(error).lower()
    triggers = (
        'rate limit',
        'rate_limit',
        'request_limit',
        'quota',
        'too many requests',
        'limit reached',
        'exceeded',
        '429',
        'insufficient credits',
        'credit limit',
        'credits',
    )
    return any(trigger in message for trigger in triggers)


def _filter_disallowed_recommendations(recommendations: list[str]) -> list[str]:
    return [item for item in recommendations if not _contains_disallowed_library(item)]


def _contains_disallowed_library(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in _DISALLOWED_LIBRARIES)


def _is_constraints_failure(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    if not _contains_disallowed_library(lowered):
        return False
    return any(token in lowered for token in ('disallow', 'not allowed', 'cannot proceed', 'constraint'))


async def _emit_phase_failure(
    *, state: MissionState, emitter: EventEmitter, phase: str, error: Exception, context: str
) -> None:
    category, strategy = classify_error(error)
    error_message = str(error)
    await emitter.emit_phase_error(phase=phase, error=error_message, recoverable=category != 'fatal')
    await emitter.emit_error(
        error_id=f'{phase}_{state.mission_id}',
        category=category,
        error_type=type(error).__name__,
        message=error_message,
        context=context,
        recovery_strategy=strategy,
    )


def _load_target_values(train_path: Path, target_column: str) -> tuple[list[float], list[str], dict[str, int] | None]:
    with train_path.open('r', encoding='utf-8', errors='ignore', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return [], [], None

        resolved_column = target_column if target_column in reader.fieldnames else reader.fieldnames[-1]
        raw_values: list[str] = []
        numeric_values: list[float] = []
        numeric = True

        for row in reader:
            raw = row.get(resolved_column, '') or ''
            raw_values.append(raw)
            try:
                numeric_values.append(float(raw))
            except (TypeError, ValueError):
                numeric = False

        if numeric:
            return numeric_values, raw_values, None

        mapping = {label: idx for idx, label in enumerate(sorted(set(raw_values)))}
        numeric_values = [float(mapping.get(value, 0)) for value in raw_values]
        return numeric_values, raw_values, mapping


def _prediction_value(
    metric: EvaluationMetric, numeric_values: list[float], raw_values: list[str], mapping: dict[str, int] | None
) -> tuple[Any, float]:
    if not numeric_values:
        return 0.0, 0.0

    mean_value = sum(numeric_values) / len(numeric_values)
    proba_metrics = {EvaluationMetric.AUC, EvaluationMetric.LOG_LOSS}
    classification_metrics = {
        EvaluationMetric.ACCURACY,
        EvaluationMetric.AUC,
        EvaluationMetric.LOG_LOSS,
        EvaluationMetric.F1,
    }

    if metric in proba_metrics:
        pred_numeric = min(max(mean_value, 1e-3), 1 - 1e-3)
        return pred_numeric, pred_numeric

    if metric in classification_metrics:
        if mapping:
            majority = Counter(raw_values).most_common(1)[0][0]
            return majority, float(mapping.get(majority, 0))
        pred_numeric = 1.0 if mean_value >= 0.5 else 0.0
        return pred_numeric, pred_numeric

    return mean_value, mean_value


def _fitness_from_score(score: float | None, direction: str) -> float | None:
    if score is None:
        return None
    value = max(score, 0.0)
    return 1.0 / (1.0 + value) if direction == 'minimize' else value


def _evaluate_metric(metric: EvaluationMetric, values: list[float], prediction: float) -> float:
    if not values:
        return 0.0

    if metric == EvaluationMetric.ACCURACY:
        correct = values.count(prediction)
        return correct / len(values)

    if metric == EvaluationMetric.F1:
        positives = values.count(1)
        negatives = len(values) - positives
        if prediction == 1:
            tp = positives
            fp = negatives
            fn = 0
        else:
            tp = 0
            fp = 0
            fn = positives
        if tp == 0:
            return 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    if metric == EvaluationMetric.AUC:
        return 0.5

    if metric == EvaluationMetric.LOG_LOSS:
        prob = min(max(prediction, 1e-6), 1 - 1e-6)
        return -sum(value * math.log(prob) + (1 - value) * math.log(1 - prob) for value in values) / len(values)
    if metric == EvaluationMetric.RMSE:
        mse = sum((value - prediction) ** 2 for value in values) / len(values)
        return math.sqrt(mse)

    if metric == EvaluationMetric.MAE:
        return sum(abs(value - prediction) for value in values) / len(values)

    if metric == EvaluationMetric.RMSLE:
        pred = max(prediction, 0.0)
        valid_values = [value for value in values if value >= 0]
        if not valid_values:
            return 0.0
        mse = sum((math.log1p(value) - math.log1p(pred)) ** 2 for value in valid_values) / len(valid_values)
        return math.sqrt(mse)

    return 0.0


def _compute_baseline_score(*, train_path: Path, target_columns: list[str], metric: EvaluationMetric) -> float:
    if not target_columns:
        return 0.0

    total_score = 0.0
    for column in target_columns:
        numeric_values, raw_values, mapping = _load_target_values(train_path, column)
        _, prediction = _prediction_value(metric, numeric_values, raw_values, mapping)
        total_score += _evaluate_metric(metric, numeric_values, prediction)

    return total_score / len(target_columns)


def _normalize_label(label: Any) -> str:
    return str(label).lower().replace('class_', '').replace('class ', '')


def _write_fallback_submission(
    *, train_path: Path, test_path: Path, sample_path: Path, metric: EvaluationMetric, output_path: Path
) -> None:
    schema = infer_competition_schema(train_path, test_path, sample_path)
    target_columns = schema.target_columns
    train_target_columns = schema.train_target_columns

    per_column_predictions: dict[str, Any] = {}
    for column in train_target_columns:
        numeric_values, raw_values, mapping = _load_target_values(train_path, column)
        prediction, _ = _prediction_value(metric, numeric_values, raw_values, mapping)
        per_column_predictions[column] = prediction

    class_probabilities: dict[str, float] = {}
    if len(train_target_columns) == 1 and len(target_columns) > 1:
        numeric_values, raw_values, _mapping = _load_target_values(train_path, train_target_columns[0])
        if raw_values:
            counts = Counter(raw_values)
            total = sum(counts.values())
            if total > 0:
                class_probabilities = {_normalize_label(label): count / total for label, count in counts.items()}

    with test_path.open('r', encoding='utf-8', errors='ignore', newline='') as handle:
        reader = csv.DictReader(handle)
        with output_path.open('w', encoding='utf-8', newline='') as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=[schema.id_column] + target_columns)
            writer.writeheader()
            for idx, row in enumerate(reader):
                row_id = row.get(schema.id_column) or str(idx)
                entry = {schema.id_column: row_id}
                if class_probabilities:
                    matched = {target: class_probabilities.get(_normalize_label(target)) for target in target_columns}
                    present = {k: v for k, v in matched.items() if v is not None}
                    remaining = max(0.0, 1.0 - sum(present.values()))
                    missing = [k for k, v in matched.items() if v is None]
                    fallback = remaining / len(missing) if missing else 0.0
                    for target in target_columns:
                        entry[target] = present.get(target, fallback)
                else:
                    for target in target_columns:
                        entry[target] = per_column_predictions.get(
                            target, next(iter(per_column_predictions.values()), 0.0)
                        )
                writer.writerow(entry)


def _generate_fallback_prototype(
    *, target_columns: list[str], train_target_columns: list[str], id_column: str, metric: EvaluationMetric
) -> str:
    metric_value = metric.value
    target_columns_repr = repr(target_columns)
    train_target_columns_repr = repr(train_target_columns)
    return (
        dedent(
            f"""
    import csv
    import math
    from collections import Counter
    
    TARGET_COLUMNS = {target_columns_repr}
    TRAIN_TARGET_COLUMNS = {train_target_columns_repr}
    ID_COLUMN = "{id_column}"
    METRIC = "{metric_value}"
    METRIC_KEY = METRIC.lower().replace("_", "")
    
    def _load_targets(path, column):
        values = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return values
            resolved = column if column in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                value = row.get(resolved, "") or ""
                values.append(value)
        return values
    
    def _numeric(values):
        numeric = []
        for value in values:
            try:
                numeric.append(float(value))
            except ValueError:
                return None
        return numeric
    
    def _prediction(values):
        numeric = _numeric(values)
        if numeric is None:
            return Counter(values).most_common(1)[0][0]
        mean_value = sum(numeric) / max(len(numeric), 1)
        if METRIC_KEY in ("auc", "logloss"):
            return min(max(mean_value, 1e-3), 1 - 1e-3)
        if METRIC_KEY in ("accuracy", "f1"):
            return 1.0 if mean_value >= 0.5 else 0.0
        return mean_value

    def _normalize_label(label):
        return str(label).lower().replace("class_", "").replace("class ", "")
    
    predictions = {{}}
    for column in TRAIN_TARGET_COLUMNS:
        targets = _load_targets("train.csv", column)
        predictions[column] = _prediction(targets)
    
    class_probs = {{}}
    if len(TRAIN_TARGET_COLUMNS) == 1 and len(TARGET_COLUMNS) > 1:
        targets = _load_targets("train.csv", TRAIN_TARGET_COLUMNS[0])
        counts = Counter(targets)
        total = sum(counts.values())
        if total > 0:
            class_probs = {{
                _normalize_label(label): count / total for label, count in counts.items()
            }}
    
    with open("test.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        with open("submission.csv", "w", newline="", encoding="utf-8") as output:
            fieldnames = [ID_COLUMN] + TARGET_COLUMNS
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for idx, row in enumerate(reader):
                row_id = row.get(ID_COLUMN) or str(idx)
                entry = {{ID_COLUMN: row_id}}
                if class_probs:
                    matched = {{
                        column: class_probs.get(_normalize_label(column))
                        for column in TARGET_COLUMNS
                    }}
                    present = {{k: v for k, v in matched.items() if v is not None}}
                    remaining = max(0.0, 1.0 - sum(present.values()))
                    missing = [k for k, v in matched.items() if v is None]
                    fallback = remaining / len(missing) if missing else 0.0
                    for column in TARGET_COLUMNS:
                        entry[column] = present.get(column, fallback)
                else:
                    for column in TARGET_COLUMNS:
                        entry[column] = predictions.get(column, next(iter(predictions.values()), 0.0))
                writer.writerow(entry)
    
    print("Baseline {metric_value} score: 0.0")
    """
        ).strip()
        + '\n'
    )


def _build_leaderboard_analysis(
    entries: list[Any], target_percentile: float, metric_direction: str
) -> LeaderboardAnalysis | None:
    scores = [entry.score for entry in entries if getattr(entry, 'score', None) is not None]
    if not scores:
        return None

    sorted_scores = sorted(scores)
    total = len(sorted_scores)
    ranked = sorted_scores[::-1] if metric_direction == 'maximize' else sorted_scores
    target_rank = max(1, math.ceil(target_percentile * total))
    target_score = ranked[target_rank - 1]

    distribution = [
        {'percentile': percentile, 'score': sorted_scores[min(total - 1, int(percentile * (total - 1)))]}
        for percentile in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]

    return LeaderboardAnalysis(
        top_score=max(scores),
        median_score=sorted_scores[total // 2],
        target_score=target_score,
        target_percentile=target_percentile,
        total_teams=total,
        score_distribution=distribution,
        common_approaches=[],
        improvement_opportunities=[],
    )


def _serialize_findings(items: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, 'model_dump'):
            results.append(item.model_dump())
        elif isinstance(item, dict):
            results.append(dict(item))
        else:
            results.append(
                {
                    'title': getattr(item, 'title', str(item)),
                    'summary': getattr(item, 'summary', ''),
                    'category': getattr(item, 'category', 'unknown'),
                    'relevance_score': getattr(item, 'relevance_score', 0.0),
                    'sources': getattr(item, 'sources', []),
                }
            )
    return results


def _looks_like_paper(finding: dict[str, Any]) -> bool:
    sources = finding.get('sources', []) or []
    return any('arxiv' in str(source).lower() or 'paper' in str(source).lower() for source in sources)


def _build_research_findings(report: Any, analysis: LeaderboardAnalysis | None) -> ResearchFindings:
    domain = _serialize_findings(getattr(report, 'domain_findings', []))
    techniques = _serialize_findings(getattr(report, 'technique_findings', []))
    findings = domain + techniques

    papers = [finding for finding in findings if _looks_like_paper(finding)]
    approaches = [finding for finding in findings if finding not in papers] or findings

    strategies = list(getattr(report, 'recommended_approaches', []) or [])
    if not strategies and approaches:
        strategies = [
            approach.get('title') or approach.get('summary', '')
            for approach in approaches
            if approach.get('title') or approach.get('summary')
        ]

    if analysis is not None:
        analysis = LeaderboardAnalysis(
            top_score=analysis.top_score,
            median_score=analysis.median_score,
            target_score=analysis.target_score,
            target_percentile=analysis.target_percentile,
            total_teams=analysis.total_teams,
            score_distribution=analysis.score_distribution,
            common_approaches=list(getattr(report, 'recommended_approaches', []) or []),
            improvement_opportunities=list(getattr(report, 'key_challenges', []) or []),
        )

    return ResearchFindings(
        leaderboard_analysis=analysis,
        papers=papers,
        approaches=approaches,
        eda_results=None,
        strategy_recommendations=strategies,
    )


def _convert_generation_history(history: list[Any], population_size: int) -> list[GenerationMetrics]:
    metrics: list[GenerationMetrics] = []
    for idx, entry in enumerate(history):
        if isinstance(entry, GenerationMetrics):
            metrics.append(entry)
            continue

        if not isinstance(entry, dict):
            continue

        metrics.append(
            GenerationMetrics(
                generation=int(entry.get('generation', idx)),
                best_fitness=float(entry.get('best_fitness', 0.0)),
                mean_fitness=float(entry.get('mean_fitness', 0.0)),
                worst_fitness=float(entry.get('worst_fitness', 0.0)),
                population_size=int(entry.get('population_size', population_size)),
                mutations=dict(entry.get('mutations', {})),
            )
        )
    return metrics
