from pathlib import Path

from config import AppConfig
from llm_refiner import (
    LLMRefiner,
    _local_filler_translation,
    _translation_num_predict_for,
    build_translation_prompt,
)
from translation_cache import TranslationCache


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_kwargs):
        self.calls += 1
        return {"response": "我们明天开始。"}


def test_local_filler_translation_handles_common_german_fillers() -> None:
    assert _local_filler_translation("Ja, genau.") == "对，没错。"
    assert _local_filler_translation("ähm") == "嗯。"


def test_local_filler_translation_ignores_real_sentence() -> None:
    assert _local_filler_translation("Wir starten morgen mit der Migration.") is None


def test_translation_num_predict_is_smaller_for_short_sentences() -> None:
    assert _translation_num_predict_for("Genau.", 128) == 64
    assert _translation_num_predict_for("Wir koennen die Datenqualitaet heute Abend pruefen.", 128) == 96
    assert _translation_num_predict_for("x" * 100, 128) == 128


def test_refiner_builds_dynamic_glossary_prompt(tmp_path: Path) -> None:
    (tmp_path / "terms.yaml").write_text(
        """
- source: Kundentabelle
  variants: ["customer table"]
  zh: 客户表
  category: data
  priority: high
""",
        encoding="utf-8",
    )
    config = AppConfig(
        meeting_profile="de",
        profile_terms_dir=tmp_path,
        custom_terms_file=tmp_path / "custom_terms.txt",
    )

    prompt = LLMRefiner(config).glossary_prompt_for_text("Bitte pruefen Sie die Kundentabelle.")

    assert "Kundentabelle => 客户表" in prompt
    assert "category: data" in prompt


def test_translation_prompt_omits_empty_glossary_section() -> None:
    prompt = build_translation_prompt(
        config=AppConfig(meeting_profile="de"),
        text="Wir starten morgen.",
        context="",
        language_code="de",
        glossary_prompt="",
        fast=True,
    )

    assert "Relevant glossary terms" not in prompt
    assert "Text:\nWir starten morgen." in prompt


def test_translation_prompt_includes_dynamic_glossary_section() -> None:
    prompt = build_translation_prompt(
        config=AppConfig(meeting_profile="de"),
        text="Bitte pruefen Sie die Kundentabelle.",
        context="",
        language_code="de",
        glossary_prompt="- Kundentabelle => 客户表 (category: data; priority: high)",
        fast=True,
    )

    assert "Relevant glossary terms" in prompt
    assert "Kundentabelle => 客户表" in prompt


def test_refiner_reuses_translation_cache(tmp_path: Path) -> None:
    config = AppConfig(
        meeting_profile="de",
        translation_style="meeting",
        translation_cache_file=tmp_path / "translation_cache.jsonl",
        structured_glossary_enabled=False,
    )
    client = FakeOllamaClient()
    cache = TranslationCache(path=config.translation_cache_file, persist=True)
    refiner = LLMRefiner(config, translation_cache=cache)
    refiner.client = client  # type: ignore[assignment]

    first = refiner.refine_and_translate_sync("Wir starten morgen.", language_code="de")
    second = refiner.refine_and_translate_sync("Wir starten morgen.", language_code="de")

    assert first == "我们明天开始。"
    assert second == "我们明天开始。"
    assert client.calls == 1
    assert refiner.last_translation_cache_hit is True
    assert refiner.last_translation_cache_source == "cache"
    assert config.translation_cache_file.exists()
