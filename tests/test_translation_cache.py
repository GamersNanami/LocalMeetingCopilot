from pathlib import Path

from translation_cache import (
    TranslationCache,
    glossary_hash,
    make_translation_cache_key,
    normalise_cache_text,
)


def test_translation_cache_memory_hit() -> None:
    cache = TranslationCache(persist=False)
    key = make_translation_cache_key(
        text="Ja genau.",
        profile="de",
        style="meeting",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="",
    )

    cache.put(key, "对，没错。")
    hit = cache.get(key)

    assert hit is not None
    assert hit.translated_text == "对，没错。"
    assert hit.source == "memory"


def test_translation_cache_persists_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "translation_cache.jsonl"
    key = make_translation_cache_key(
        text="Wir starten morgen.",
        profile="de",
        style="meeting",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="",
    )
    TranslationCache(path=path, persist=True).put(key, "我们明天开始。")

    hit = TranslationCache(path=path, persist=True).get(key)

    assert hit is not None
    assert hit.translated_text == "我们明天开始。"
    assert hit.source == "disk"


def test_translation_cache_respects_ttl(tmp_path: Path) -> None:
    now = 1000.0
    path = tmp_path / "translation_cache.jsonl"
    key = make_translation_cache_key(
        text="Rollout",
        profile="de",
        style="meeting",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="",
    )
    cache = TranslationCache(path=path, persist=True, ttl_seconds=10, now=lambda: now)
    cache.put(key, "发布")

    expired_cache = TranslationCache(path=path, persist=True, ttl_seconds=10, now=lambda: now + 11)

    assert expired_cache.get(key) is None


def test_translation_cache_prunes_old_entries() -> None:
    counter = {"now": 0.0}

    def now() -> float:
        counter["now"] += 1.0
        return counter["now"]

    cache = TranslationCache(max_entries=2, now=now)
    for text in ("one", "two", "three"):
        key = make_translation_cache_key(
            text=text,
            profile="de",
            style="meeting",
            model="qwen2.5:3b-instruct",
            language_code="de",
            glossary_prompt="",
        )
        cache.put(key, text)

    assert len(cache) == 2


def test_translation_cache_key_separates_style_and_glossary() -> None:
    left = make_translation_cache_key(
        text="Kundentabelle",
        profile="de",
        style="meeting",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="- Kundentabelle => 客户表",
    )
    right = make_translation_cache_key(
        text="Kundentabelle",
        profile="de",
        style="literal",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="- Kundentabelle => 客户表",
    )
    changed_glossary = make_translation_cache_key(
        text="Kundentabelle",
        profile="de",
        style="meeting",
        model="qwen2.5:3b-instruct",
        language_code="de",
        glossary_prompt="- Kundentabelle => 客户名单表",
    )

    assert left.digest != right.digest
    assert left.digest != changed_glossary.digest


def test_normalise_cache_text_handles_german_spacing() -> None:
    assert normalise_cache_text("Daten-Qualität!") == "daten qualitaet"
    assert glossary_hash("- Rollout => 发布") == glossary_hash(" -  Rollout   => 发布 ")
