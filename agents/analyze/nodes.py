"""
agents/analyze/nodes.py — Public exports for Analyze graph nodes.

Implementation split across nodes_*.py modules; this barrel preserves
backward-compatible imports for graph.py, tests, and evals.
"""
from agents.analyze.nodes_cache import (
    check_semantic_cache,
    semantic_cache_gate,
    inject_history,
    load_schema,
)
from agents.analyze.nodes_intent import (
    infer_metric_config_node,
    resolve_task_intent,
    _apply_intent_to_config,
)
from agents.analyze.nodes_sql import (
    execute_query,
    generate_sql,
    query_gate,
)
from agents.analyze.nodes_analysis import (
    analysis_gate,
    check_guardrails_node,
    check_srm_node,
    compute_funnel_node,
    compute_mde_node,
    decompose_metric,
    describe_data_node,
    detect_anomaly_node,
    detect_novelty_node,
    detect_timeseries_node,
    find_correlations_node,
    forecast_baseline_node,
    generate_charts_node,
    load_auxiliary_data,
    run_cuped_node,
    run_hte_node,
    run_power_analysis_node,
    run_regression_node,
    run_ttest_node,
)
from agents.analyze.nodes_narrative import (
    generate_narrative,
    log_run_node,
    narrative_gate,
)
from agents.analyze.node_shared import (
    _build_few_shot_block,
    _canonical_experiment_sql,
    _columns_for_table,
    _db_conn,
    _extract_sql,
    _filter_few_shot_by_schema,
    _known_schema_names,
    _sanitise_metric_config,
    _tables_in_sql,
    _validate_sql_references,
)

__all__ = [
    "check_semantic_cache", "semantic_cache_gate", "inject_history", "load_schema",
    "resolve_task_intent", "infer_metric_config_node",
    "generate_sql", "query_gate", "execute_query",
    "load_auxiliary_data", "decompose_metric", "detect_anomaly_node", "forecast_baseline_node",
    "run_cuped_node", "run_ttest_node", "check_srm_node", "run_hte_node", "detect_novelty_node",
    "compute_mde_node", "run_power_analysis_node", "check_guardrails_node", "compute_funnel_node",
    "describe_data_node", "find_correlations_node", "run_regression_node", "detect_timeseries_node",
    "generate_charts_node", "analysis_gate",
    "generate_narrative", "narrative_gate", "log_run_node",
    "_apply_intent_to_config",
    "_build_few_shot_block", "_filter_few_shot_by_schema", "_known_schema_names",
    "_columns_for_table", "_tables_in_sql",
    "_canonical_experiment_sql", "_sanitise_metric_config", "_validate_sql_references",
    "_extract_sql", "_db_conn",
]
