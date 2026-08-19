import numpy as np

from asr_engine import (
    ASREngine,
    _language_allowed,
    _looks_like_hallucination,
    prepare_audio_for_asr,
)
from config import AppConfig


def test_prepare_audio_for_asr_removes_dc_and_limits_peak() -> None:
    audio = np.array([4.0, 5.0, 6.0, 5.0], dtype=np.float32)

    prepared = prepare_audio_for_asr(audio, sample_rate=16_000)

    assert prepared.dtype == np.float32
    assert abs(float(prepared.mean())) < 0.1
    assert float(np.max(np.abs(prepared))) <= 1.0


def test_language_allowed_rejects_arabic() -> None:
    assert _language_allowed("de", ("de", "en", "zh"))
    assert not _language_allowed("ar", ("de", "en", "zh"))


def test_hallucination_filter_blocks_common_subtitle_phrase() -> None:
    assert _looks_like_hallucination("Thanks for watching", ("thanks for watching",))
    assert not _looks_like_hallucination("Wir starten morgen.", ("thanks for watching",))


def test_transcribe_sentence_retries_disallowed_language() -> None:
    class FakeSegment:
        text = "Wir starten morgen."
        no_speech_prob = 0.1

    class FakeInfo:
        def __init__(self, language: str) -> None:
            self.language = language

    class FakeModel:
        def __init__(self) -> None:
            self.languages: list[str | None] = []
            self.kwargs: list[dict[str, object]] = []

        def transcribe(self, _audio, **kwargs):
            self.languages.append(kwargs["language"])
            self.kwargs.append(kwargs)
            if len(self.languages) == 1:
                return [FakeSegment()], FakeInfo("ar")
            return [FakeSegment()], FakeInfo("de")

    engine = ASREngine(AppConfig(audio_sample_rate=16_000, asr_min_audio_seconds=0.1))
    fake_model = FakeModel()
    engine._model = fake_model
    audio = np.ones(16_000, dtype=np.float32) * 0.05

    language, text, confidence = engine.transcribe_sentence(audio)

    assert fake_model.languages == [None, "de"]
    assert fake_model.kwargs[0]["initial_prompt"]
    assert fake_model.kwargs[0]["hotwords"]
    assert language == "de"
    assert text == "Wir starten morgen."
    assert confidence is not None


def test_partial_transcribe_uses_partial_settings() -> None:
    class FakeSegment:
        text = "Wir prüfen das"
        no_speech_prob = 0.1

    class FakeInfo:
        language = "de"

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, object]] = []

        def transcribe(self, _audio, **kwargs):
            self.kwargs.append(kwargs)
            return [FakeSegment()], FakeInfo()

    engine = ASREngine(
        AppConfig(
            audio_sample_rate=16_000,
            partial_min_audio_seconds=0.1,
            partial_asr_beam_size=1,
            partial_asr_vad_filter=False,
        )
    )
    fake_model = FakeModel()
    engine._model = fake_model
    audio = np.ones(16_000, dtype=np.float32) * 0.05

    language, text, _confidence = engine.transcribe_sentence(audio, partial=True)

    assert language == "de"
    assert text == "Wir prüfen das"
    assert fake_model.kwargs[0]["beam_size"] == 1
    assert fake_model.kwargs[0]["vad_filter"] is False


def test_german_profile_transcribes_with_forced_language() -> None:
    class FakeSegment:
        text = "Das passt."
        no_speech_prob = 0.1

    class FakeInfo:
        language = "de"

    class FakeModel:
        def __init__(self) -> None:
            self.languages: list[str | None] = []

        def transcribe(self, _audio, **kwargs):
            self.languages.append(kwargs["language"])
            return [FakeSegment()], FakeInfo()

    engine = ASREngine(AppConfig(meeting_profile="de", asr_allowed_languages=("de",), asr_force_language="de"))
    fake_model = FakeModel()
    engine._model = fake_model
    audio = np.ones(16_000, dtype=np.float32) * 0.05

    language, text, _confidence = engine.transcribe_sentence(audio)

    assert fake_model.languages == ["de"]
    assert language == "de"
    assert text == "Das passt."
