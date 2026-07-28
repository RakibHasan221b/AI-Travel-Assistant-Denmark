import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "llm" / "prompts"))
from rag_summary import RAG_SUMMARY_PROMPT

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def test_ai_summaries_table_defined():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "CREATE TABLE ai_summaries" in sql


def test_rag_summary_prompt_formats_with_dummy_inputs():
    formatted = RAG_SUMMARY_PROMPT.format(
        place_name="Test Cafe",
        category="cafe",
        neighborhood="Vesterbro",
        aspect_facts="- food: 4.0/5 (2 mention(s))",
        snippets="[1] (wikivoyage) A cozy spot with good coffee.",
    )
    assert "Test Cafe" in formatted
    assert "cozy spot" in formatted
