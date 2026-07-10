import os

import pytest

from tools.db_tools import DBConnection


pytestmark = pytest.mark.integration


def _pg_config() -> dict[str, object]:
    if os.getenv("DATAPILOT_POSTGRES_INTEGRATION") != "1":
        pytest.skip("DATAPILOT_POSTGRES_INTEGRATION=1 is not configured")
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "datapilot_test"),
        "user": os.getenv("PGUSER", "datapilot"),
        "password": os.getenv("PGPASSWORD", "datapilot"),
    }


def test_postgres_query_schema_and_readonly_guards():
    psycopg2 = pytest.importorskip("psycopg2")
    cfg = _pg_config()

    admin = psycopg2.connect(**cfg)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS datapilot_ci_metrics")
            cur.execute("""
                CREATE TABLE datapilot_ci_metrics (
                    day date PRIMARY KEY,
                    metric integer NOT NULL,
                    segment text NOT NULL
                )
            """)
            cur.execute("""
                INSERT INTO datapilot_ci_metrics(day, metric, segment)
                VALUES ('2026-01-01', 10, 'control'),
                       ('2026-01-02', 12, 'treatment')
            """)
    finally:
        admin.close()

    db = DBConnection("postgres", **cfg)
    df = db.query("""
        /* leading comments are allowed */
        SELECT segment, SUM(metric) AS total_metric
        FROM datapilot_ci_metrics
        GROUP BY segment
        ORDER BY segment
    """)

    assert df["segment"].tolist() == ["control", "treatment"]
    assert df["total_metric"].tolist() == [10, 12]

    with pytest.raises(ValueError):
        db.query("CREATE TABLE should_not_run(id integer)")

    schema = db.inspect_schema()
    assert "TABLE: datapilot_ci_metrics" in schema
    assert "metric" in schema
