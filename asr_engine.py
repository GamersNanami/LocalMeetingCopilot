from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

from config import AppConfig, load_config
from meeting_types import TranscriptDraft


class ASREngine:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._model = None
        self._lock = Lock()

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.config.asr_model_size,
                device=self.config.asr_device,
                compute_type=self.config.asr_compute_type,
                download_root=str(self.config.model_cache_dir),
            )
        return self._model

    def warm_up(self) -> None:
        with self._lock:
            _ = self.model

    def transcribe_stream_chunk(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
        return "..." if rms > 0.01 else ""

    def transcribe_sentence(self, audio: np.ndarray, *, partial: bool = False) -> tuple[str, str, float | None]:
        prepared = prepare_audio_for_asr(audio, self.config.audio_sample_rate)
        min_seconds = self.config.partial_min_audio_seconds if partial else self.config.asr_min_audio_seconds
        if prepared.size < int(min_seconds * self.config.audio_sample_rate):
            return "auto", "", None

        beam_size = self.config.partial_asr_beam_size if partial else self.config.asr_beam_size
        vad_filter = self.config.partial_asr_vad_filter if partial else True
        with self._lock:
            segments, info = self._transcribe_prepared(
                prepared,
                beam_size=beam_size,
                language=self.config.asr_force_language,
                vad_filter=vad_filter,
            )
            if self._should_retry_language(info.language):
                segments, info = self._transcribe_prepared(
                    prepared,
                    beam_size=beam_size,
                    language=self.config.asr_default_language,
                    vad_filter=vad_filter,
                )

        language = _normalise_language(info.language)
        if not _language_allowed(language, self.config.asr_allowed_languages):
            return language or "auto", "", _confidence_from_segments(segments)

        text = " ".join(segment.text.strip() for segment in segments).strip()
        if _looks_like_hallucination(text, self.config.asr_hallucination_phrases):
            return language or "auto", "", _confidence_from_segments(segments)
        return language or "auto", text, _confidence_from_segments(segments)

    def transcribe_file(self, filepath: str | Path, speaker: str = "Remote Participant") -> list[TranscriptDraft]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)

        with self._lock:
            segments, info = self._transcribe_file_path(
                str(path),
                beam_size=self.config.asr_file_beam_size,
                language=self.config.asr_force_language,
            )
            if self._should_retry_language(info.language):
                segments, info = self._transcribe_file_path(
                    str(path),
                    beam_size=self.config.asr_file_beam_size,
                    language=self.config.asr_default_language,
                )

        language = _normalise_language(info.language)
        if not _language_allowed(language, self.config.asr_allowed_languages):
            return []

        return [
            TranscriptDraft(
                speaker=speaker,
                text=segment.text.strip(),
                language_code=language or "auto",
                track_type="wav",
                start_time=float(segment.start),
                end_time=float(segment.end),
                confidence=_confidence_from_segments([segment]),
            )
            for segment in segments
            if segment.text.strip()
        ]

    def _transcribe_prepared(
        self,
        prepared: np.ndarray,
        *,
        beam_size: int,
        language: str | None,
        vad_filter: bool,
    ) -> tuple[list[object], SimpleNamespace]:
        segments, info = self.model.transcribe(
            prepared,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
            condition_on_previous_text=self.config.asr_condition_on_previous_text,
            initial_prompt=self.config.asr_initial_prompt,
            hotwords=self.config.asr_hotwords,
        )
        return list(segments), SimpleNamespace(language=getattr(info, "language", None))

    def _transcribe_file_path(
        self,
        path: str,
        *,
        beam_size: int,
        language: str | None,
    ) -> tuple[list[object], SimpleNamespace]:
        segments, info = self.model.transcribe(
            path,
            beam_size=beam_size,
            vad_filter=True,
            word_timestamps=False,
            language=language,
            condition_on_previous_text=self.config.asr_condition_on_previous_text,
            initial_prompt=self.config.asr_initial_prompt,
            hotwords=self.config.asr_hotwords,
        )
        return list(segments), SimpleNamespace(language=getattr(info, "language", None))

    def _should_retry_language(self, language: str | None) -> bool:
        if self.config.asr_force_language:
            return False
        if not self.config.asr_retry_disallowed_language:
            return False
        normalised = _normalise_language(language)
        return not _language_allowed(normalised, self.config.asr_allowed_languages)


def drafts_from_texts(texts: Iterable[str], speaker: str = "Remote Participant") -> list[TranscriptDraft]:
    return [
        TranscriptDraft(
            speaker=speaker,
            text=text,
            language_code="auto",
            track_type="mock",
            start_time=float(index * 3),
            end_time=float(index * 3 + 2),
        )
        for index, text in enumerate(texts)
    ]


def prepare_audio_for_asr(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    mono = mono - float(np.mean(mono))
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        mono = mono / peak
    if sample_rate > 0:
        trim_samples = min(int(0.08 * sample_rate), mono.size // 8)
        if trim_samples > 0:
            mono[:trim_samples] *= np.linspace(0.0, 1.0, trim_samples, dtype=np.float32)
            mono[-trim_samples:] *= np.linspace(1.0, 0.0, trim_samples, dtype=np.float32)
    return np.ascontiguousarray(mono)


def _normalise_language(language: str | None) -> str:
    if not language:
        return "auto"
    normalised = language.lower().strip()
    if normalised.startswith("zh"):
        return "zh"
    return normalised


def _language_allowed(language: str, allowed_languages: Iterable[str]) -> bool:
    allowed = {_normalise_language(item) for item in allowed_languages}
    return language in allowed


def _looks_like_hallucination(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    compact = " ".join(lowered.split())
    if len(compact) <= 3:
        return True
    return any(phrase.lower() in compact for phrase in phrases)


def _confidence_from_segments(segments: Iterable[object]) -> float | None:
    values: list[float] = []
    for segment in segments:
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        avg_logprob = getattr(segment, "avg_logprob", None)
        if no_speech_prob is not None:
            values.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
        elif avg_logprob is not None:
            values.append(max(0.0, min(1.0, 1.0 + float(avg_logprob))))
    if not values:
        return None
    return sum(values) / len(values)
