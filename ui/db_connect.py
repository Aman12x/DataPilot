"""
ui/db_connect.py — Streamlit sidebar component for database connection and
MetricConfig setup.

Two sections rendered in a single sidebar expander:
  1. Database connection (DuckDB default / Postgres form with test button)
  2. MetricConfig form (pre-filled from JSON; analyst confirms before first run)

Rules:
  - This file only renders UI. No stats, SQL, or agent logic.
  - Postgres credentials are stored only in st.session_state for the session
    duration — never written to disk, SQLite, or logs.
  - DB connection is tested before being accepted.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from config.analysis_config import MetricConfig, load_metric_config
from tools.db_tools import DBConnection


# ── Internal helpers ──────────────────────────────────────────────────────────

def _reset_schema_cache() -> None:
    """Delete the schema cache so load_schema re-inspects on the next run."""
    cache_path = os.getenv("SCHEMA_CACHE_PATH", "memory/schema_cache.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)


def _metric_config_form(defaults: MetricConfig) -> MetricConfig | None:
    """
    Render a MetricConfig form pre-filled from `defaults`.
    Returns the confirmed MetricConfig, or None if not yet confirmed.
    """
    st.caption(
        "Review the metric settings inferred from the schema. "
        "Edit any field before running the first analysis."
    )

    with st.form("metric_config_form"):
        primary_metric = st.text_input(
            "Primary metric column",
            value=defaults.primary_metric,
            help="The outcome column the experiment is measured on.",
        )
        covariate = st.text_input(
            "CUPED covariate column",
            value=defaults.covariate,
            help="Pre-experiment column correlated with the primary metric.",
        )
        metric_direction = st.selectbox(
            "Metric direction",
            options=["higher_is_better", "lower_is_better"],
            index=0 if defaults.metric_direction == "higher_is_better" else 1,
            help="Whether higher values are good (e.g. DAU) or bad (e.g. churn rate).",
        )
        guardrail_metrics = st.text_input(
            "Guardrail metrics (comma-separated)",
            value=", ".join(defaults.guardrail_metrics),
            help="Secondary metrics to monitor for regressions.",
        )
        segment_cols = st.text_input(
            "Segment columns (comma-separated)",
            value=", ".join(defaults.segment_cols),
            help="Dimension columns used for HTE and anomaly slice-and-dice.",
        )
        funnel_steps = st.text_input(
            "Funnel steps (comma-separated, ordered)",
            value=", ".join(defaults.funnel_steps),
            help="Ordered funnel steps for conversion analysis.",
        )
        revenue_per_unit = st.number_input(
            "Revenue per unit (USD)",
            value=defaults.revenue_per_unit,
            min_value=0.0,
            step=0.01,
            format="%.2f",
            help="Used in MDE business impact statement.",
        )

        confirmed = st.form_submit_button("Confirm metric configuration")

    if not confirmed:
        return None

    def _csv(s: str) -> list[str]:
        return [v.strip() for v in s.split(",") if v.strip()]

    return MetricConfig(
        primary_metric=primary_metric.strip(),
        covariate=covariate.strip(),
        metric_direction=metric_direction,  # type: ignore[arg-type]
        guardrail_metrics=_csv(guardrail_metrics),
        segment_cols=_csv(segment_cols),
        funnel_steps=_csv(funnel_steps),
        revenue_per_unit=float(revenue_per_unit),
        experiment_weeks=defaults.experiment_weeks,
        guardrail_harm_directions=defaults.guardrail_harm_directions,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def render_connection_sidebar() -> None:
    """
    Render the database connection + MetricConfig sidebar expander.

    Writes to st.session_state:
      db_conn        — DBConnection object (cleared on session end)
      metric_config  — confirmed MetricConfig
      schema_needs_refresh — True when a new DB is connected
    """
    with st.sidebar.expander("Database & metric setup", expanded=True):
        # ── Section 1: database selection ─────────────────────────────────────
        st.markdown("#### Database connection")

        db_choice = st.radio(
            "Data source",
            options=["Built-in demo dataset (DuckDB)", "Connect my Postgres database"],
            index=0,
            key="db_choice_radio",
        )

        if db_choice == "Built-in demo dataset (DuckDB)":
            duckdb_path = os.getenv("DUCKDB_PATH", "data/dau_experiment.db")
            if st.button("Use demo dataset", key="use_duckdb_btn"):
                conn = DBConnection(backend="duckdb", path=duckdb_path)
                result = conn.test_connection()
                if result["success"]:
                    st.session_state.db_conn = conn
                    st.session_state.schema_needs_refresh = True
                    _reset_schema_cache()
                    st.success(f"Connected: {result['table_count']} tables found")
                else:
                    st.error(f"Connection failed: {result['error']}")

        else:
            # Postgres form — credentials never leave session_state
            with st.form("postgres_form"):
                host     = st.text_input("Host",     value="localhost")
                port     = st.number_input("Port",   value=5432, min_value=1, max_value=65535)
                dbname   = st.text_input("Database", value="")
                user     = st.text_input("User",     value="")
                password = st.text_input("Password", type="password", value="")
                sslmode  = st.selectbox("SSL mode",
                                        ["require", "prefer", "allow", "disable"],
                                        index=0)

                col1, col2 = st.columns(2)
                test_clicked = col1.form_submit_button("Test connection")
                use_clicked  = col2.form_submit_button("Use this database")

            if test_clicked or use_clicked:
                # Confirmation before any connection attempt (Rule 10)
                st.info(
                    f"Connecting to **{host}:{port}/{dbname}** "
                    f"(ssl={sslmode}) …"
                )
                conn = DBConnection(
                    backend="postgres",
                    host=host,
                    port=int(port),
                    dbname=dbname,
                    user=user,
                    password=password,
                    sslmode=sslmode,
                )
                result = conn.test_connection()
                if result["success"]:
                    if use_clicked:
                        st.session_state.db_conn = conn
                        st.session_state.schema_needs_refresh = True
                        _reset_schema_cache()
                        st.success(
                            f"Connected: {result['table_count']} tables found. "
                            "Schema refresh will run on next analysis."
                        )
                    else:
                        st.success(
                            f"Test passed: {result['table_count']} tables found."
                        )
                else:
                    st.error(f"Connection failed: {result['error']}")

        st.divider()

        # ── Section 2: MetricConfig ────────────────────────────────────────────
        st.markdown("#### Metric configuration")

        _PRESETS = {
            "DAU drop (default)":      "config/examples/dau_drop.json",
            "Revenue experiment":       "config/examples/revenue_experiment.json",
            "Retention experiment":     "config/examples/retention_experiment.json",
            "Custom (edit below)":      None,
        }
        preset_choice = st.selectbox(
            "Load preset",
            options=list(_PRESETS.keys()),
            key="preset_selector",
            help="Pre-fill the metric configuration from a scenario template.",
        )

        preset_path = _PRESETS[preset_choice]
        if preset_path and os.path.exists(preset_path):
            try:
                with open(preset_path) as f:
                    preset_data = json.load(f)
                preset_mc = MetricConfig(**preset_data)
            except Exception:
                preset_mc = load_metric_config()
        else:
            preset_mc = st.session_state.get("metric_config") or load_metric_config()

        confirmed_mc = _metric_config_form(defaults=preset_mc)
        if confirmed_mc is not None:
            st.session_state.metric_config = confirmed_mc
            st.success(f"Metric configuration saved ({preset_choice}).")
