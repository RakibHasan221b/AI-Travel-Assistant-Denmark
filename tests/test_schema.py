from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def test_schema_file_exists_and_defines_core_tables():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for table in ["places", "reviews_raw", "aspect_sentiment", "pipeline_runs"]:
        assert f"CREATE TABLE {table}" in sql, f"missing table: {table}"
