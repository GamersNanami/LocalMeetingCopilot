from llm_refiner import _local_filler_translation, _translation_num_predict_for


def test_local_filler_translation_handles_common_german_fillers() -> None:
    assert _local_filler_translation("Ja, genau.") == "对，没错。"
    assert _local_filler_translation("ähm") == "嗯。"


def test_local_filler_translation_ignores_real_sentence() -> None:
    assert _local_filler_translation("Wir starten morgen mit der Migration.") is None


def test_translation_num_predict_is_smaller_for_short_sentences() -> None:
    assert _translation_num_predict_for("Genau.", 128) == 64
    assert _translation_num_predict_for("Wir koennen die Datenqualitaet heute Abend pruefen.", 128) == 96
    assert _translation_num_predict_for("x" * 100, 128) == 128
