from __future__ import annotations

from pathlib import Path

from glossary import (
    format_term_matches_for_prompt,
    load_glossary_terms,
    load_structured_terms,
    match_terms,
    normalise_term_text,
)


def test_load_structured_terms_supports_yaml_list(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text(
        """
- source: Kundentabelle
  variants: ["Kunden Tabelle", "customer table"]
  zh: 客户表
  category: data
  priority: high
  profiles: ["de", "de-en"]
""",
        encoding="utf-8",
    )

    terms = load_structured_terms(path, profile="de")

    assert len(terms) == 1
    assert terms[0].source == "Kundentabelle"
    assert terms[0].variants == ("Kunden Tabelle", "customer table")
    assert terms[0].zh == "客户表"


def test_structured_terms_filter_profiles(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text(
        """
terms:
  - source: Rollout
    zh: 发布
    profiles: ["de"]
  - source: Customer table
    zh: 客户表
    profiles: ["en"]
""",
        encoding="utf-8",
    )

    terms = load_structured_terms(path, profile="de")

    assert [term.source for term in terms] == ["Rollout"]


def test_match_terms_uses_variants_and_priority(tmp_path: Path) -> None:
    terms = [
        load_structured_terms_from_text(
            tmp_path / "high.yaml",
            """
- source: Kundentabelle
  variants: ["Kunden Tabelle", "customer table"]
  zh: 客户表
  category: data
  priority: high
- source: Tabelle
  zh: 表
  category: data
  priority: low
""",
        )[0],
        load_structured_terms_from_text(
            tmp_path / "low.yaml",
            """
- source: Tabelle
  zh: 表
  category: data
  priority: low
""",
        )[0],
    ]

    matches = match_terms("Bitte pruefen Sie die Kunden Tabelle bis morgen.", terms)

    assert matches[0].term.source == "Kundentabelle"
    assert matches[0].term.zh == "客户表"


def test_match_terms_handles_german_transliteration_and_compounds(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text(
        """
- source: Datenqualität
  variants: ["Daten Qualitaet"]
  zh: 数据质量
  priority: high
""",
        encoding="utf-8",
    )
    terms = load_structured_terms(path, profile="de")

    matches = match_terms("Wir muessen die Datenqualitaet verbessern.", terms)

    assert matches[0].term.zh == "数据质量"


def test_load_glossary_terms_includes_legacy_txt(tmp_path: Path) -> None:
    (tmp_path / "de_terms.txt").write_text("Datenpipeline\n", encoding="utf-8")
    custom_terms = tmp_path / "custom_terms.txt"
    custom_terms.write_text("Musterkunde\n", encoding="utf-8")

    terms = load_glossary_terms(
        profile="de",
        profile_terms_dir=tmp_path,
        custom_terms_file=custom_terms,
    )

    assert {term.source for term in terms} >= {"Datenpipeline", "Musterkunde"}
    assert all(term.legacy for term in terms)


def test_format_term_matches_for_prompt(tmp_path: Path) -> None:
    terms = load_structured_terms_from_text(
        tmp_path / "terms.yaml",
        """
- source: Rollout
  variants: ["deployment"]
  zh: 发布
  category: release
  priority: high
""",
    )
    matches = match_terms("The deployment starts tomorrow.", terms)

    prompt = format_term_matches_for_prompt(matches)

    assert "Rollout => 发布" in prompt
    assert "variants: deployment" in prompt


def test_normalise_term_text_handles_umlauts() -> None:
    assert normalise_term_text("Datenqualität!") == "datenqualitaet"


def load_structured_terms_from_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    return load_structured_terms(path, profile="de")
