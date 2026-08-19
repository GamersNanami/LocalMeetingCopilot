from __future__ import annotations

import argparse
import asyncio
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import sounddevice
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication

from asr_engine import ASREngine
from audio_engine import (
    AudioEngine,
    _loopback_device_candidates,
    _remote_input_device_candidates,
    _select_remote_input_device,
    _select_wasapi_loopback_device,
)
from config import (
    AppConfig,
    apply_meeting_profile,
    apply_model_preset,
    load_config,
    save_runtime_settings,
)
from llm_refiner import LLMRefiner
from meeting_types import TranscriptDraft, TranscriptEntry
from summarizer import MeetingSummarizer
from ui.dashboard_window import MeetingDashboard
from ui.overlay_window import SubtitleOverlay


class WorkerSignals(QObject):
    status = Signal(str)
    draft_ready = Signal(object)
    translation_ready = Signal(object, str)
    report_ready = Signal(str)
    error = Signal(str)
    finished = Signal()


class TranslationTask(QRunnable):
    def __init__(self, config: AppConfig, draft: TranscriptDraft, context_history: list[str]) -> None:
        super().__init__()
        self.config = config
        self.draft = draft
        self.context_history = context_history
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.draft.translation_started_at = time.perf_counter()
            translated = LLMRefiner(self.config).refine_and_translate_sync(
                self.draft.text,
                self.context_history,
                language_code=self.draft.language_code,
            )
            self.draft.translation_completed_at = time.perf_counter()
            self.signals.translation_ready.emit(self.draft, translated)
        except Exception as exc:
            self.signals.error.emit(f"Translation failed: {exc}")
        finally:
            self.signals.finished.emit()


class SummaryTask(QRunnable):
    def __init__(self, config: AppConfig, transcript_markdown: str) -> None:
        super().__init__()
        self.config = config
        self.transcript_markdown = transcript_markdown
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            report = LLMRefiner(self.config).summarize_meeting_sync(self.transcript_markdown)
            self.signals.report_ready.emit(report)
        except Exception as exc:
            self.signals.error.emit(f"Summary failed: {exc}")
        finally:
            self.signals.finished.emit()


class WavTranscriptionTask(QRunnable):
    def __init__(self, config: AppConfig, asr_engine: ASREngine, wav_path: Path) -> None:
        super().__init__()
        self.config = config
        self.asr_engine = asr_engine
        self.wav_path = wav_path
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.status.emit(f"Loading Whisper model: {self.config.asr_model_size}")
            drafts = self.asr_engine.transcribe_file(self.wav_path)
            if not drafts:
                self.signals.status.emit("No speech detected in WAV file")
            for draft in drafts:
                self.signals.draft_ready.emit(draft)
        except Exception as exc:
            self.signals.error.emit(f"WAV transcription failed: {exc}")
        finally:
            self.signals.finished.emit()


class AudioTranscriptionTask(QRunnable):
    def __init__(self, config: AppConfig, asr_engine: ASREngine, draft: TranscriptDraft) -> None:
        super().__init__()
        self.config = config
        self.asr_engine = asr_engine
        self.draft = draft
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        if self.draft.audio_data is None or self.draft.audio_data.size == 0:
            self.signals.error.emit("No audio data to transcribe")
            self.signals.finished.emit()
            return

        try:
            self.signals.status.emit(f"Transcribing [{self.draft.speaker}] with Whisper")
            self.draft.asr_started_at = time.perf_counter()
            language_code, text, confidence = self.asr_engine.transcribe_sentence(self.draft.audio_data)
            self.draft.asr_completed_at = time.perf_counter()
            self.draft.language_code = language_code
            self.draft.text = text
            self.draft.confidence = confidence
            if text:
                self.signals.draft_ready.emit(self.draft)
            else:
                self.signals.status.emit("Whisper returned no text for this sentence")
        except Exception as exc:
            self.signals.error.emit(f"ASR failed: {exc}")
        finally:
            self.signals.finished.emit()


class PartialTranscriptionTask(QRunnable):
    def __init__(self, config: AppConfig, asr_engine: ASREngine, draft: TranscriptDraft) -> None:
        super().__init__()
        self.config = config
        self.asr_engine = asr_engine
        self.draft = draft
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        if self.draft.audio_data is None or self.draft.audio_data.size == 0:
            self.signals.finished.emit()
            return

        try:
            language_code, text, confidence = self.asr_engine.transcribe_sentence(
                self.draft.audio_data,
                partial=True,
            )
            self.draft.language_code = language_code
            self.draft.text = text
            self.draft.confidence = confidence
            if text:
                self.signals.draft_ready.emit(self.draft)
        except Exception as exc:
            self.signals.error.emit(f"Partial ASR failed: {exc}")
        finally:
            self.signals.finished.emit()


class MeetingAppController(QObject):
    def __init__(self, args: argparse.Namespace, config: AppConfig | None = None) -> None:
        super().__init__()
        self.args = args
        self.config = config or load_config(
            profile=args.profile,
            preset=args.preset,
            translation_style=args.style,
        )
        _apply_runtime_arg_overrides(self.config, args)
        self.thread_pool = QThreadPool.globalInstance()
        self.summarizer = MeetingSummarizer(self.config)
        self.asr_engine = ASREngine(self.config)
        self.audio_engine: AudioEngine | None = None
        self.speaker_aliases = dict(self.config.speaker_aliases)
        self._running_source = False
        self._last_status = "Ready"
        self._translation_queue: deque[TranscriptDraft] = deque()
        self._translation_busy = False
        self._partial_busy_tracks: set[str] = set()
        self._latest_ai_summary: str | None = None
        self._merge_buffer: TranscriptDraft | None = None
        self._merge_timer = QTimer(self)
        self._merge_timer.setSingleShot(True)
        self._merge_timer.timeout.connect(self._flush_merge_buffer)

        self.overlay = SubtitleOverlay(self.config)
        self.dashboard = MeetingDashboard(self.config)
        self.overlay.open_dashboard_requested.connect(self.dashboard.showNormal)
        self.overlay.close_requested.connect(QApplication.instance().quit)
        self.dashboard.start_requested.connect(self.start)
        self.dashboard.pause_requested.connect(self.pause)
        self.dashboard.end_requested.connect(self.end)
        self.dashboard.export_requested.connect(self.export_report)
        self.dashboard.summary_requested.connect(self.generate_summary)
        self.dashboard.settings_changed.connect(self._on_dashboard_settings_changed)
        self.dashboard.speaker_rename_requested.connect(self._on_speaker_rename_requested)

    def show(self) -> None:
        self.dashboard.show()
        self.overlay.show()

    @Slot()
    def start(self) -> None:
        if self._running_source:
            self._set_status("Meeting source is already running")
            return

        if self.args.wav:
            self._running_source = True
            self.dashboard.set_running_state(True)
            self._start_wav(Path(self.args.wav))
            return

        self.audio_engine = AudioEngine(self.config, mode=self.args.mode)
        self.audio_engine.preview_updated.connect(self._on_preview)
        self.audio_engine.partial_speech_ready.connect(self._on_partial_draft)
        self.audio_engine.sentence_completed.connect(self._on_draft)
        self.audio_engine.status_changed.connect(self._set_status)
        self.audio_engine.finished.connect(self._on_audio_finished)
        self._running_source = True
        self.dashboard.set_running_state(True)
        self.audio_engine.start()

    @Slot()
    def pause(self) -> None:
        if self.audio_engine:
            self.audio_engine.stop()
            self.audio_engine = None
        self._running_source = False
        self.dashboard.set_running_state(False)
        self._set_status("Paused")

    @Slot()
    def end(self) -> None:
        if self.audio_engine:
            self.audio_engine.stop()
            self.audio_engine = None
        self._flush_merge_buffer()
        self._running_source = False
        self.dashboard.set_running_state(False)
        self._set_status("Meeting ended")

    @Slot()
    def export_report(self) -> None:
        self._flush_merge_buffer()
        markdown = self.summarizer.build_markdown_report(ai_summary=self._latest_ai_summary)
        if self.config.privacy_mode or not self.config.save_reports_enabled:
            self.dashboard.set_report(markdown)
            self._set_status("Privacy mode: report preview generated without disk export")
            return

        markdown_path = self.summarizer.export_markdown(ai_summary=self._latest_ai_summary)
        self.summarizer.export_json(markdown_path.with_suffix(".json"))
        self.dashboard.set_report(markdown, markdown_path)
        self._set_status(f"Report exported: {markdown_path}")

    @Slot()
    def generate_summary(self) -> None:
        self._flush_merge_buffer()
        if not self.summarizer.entries:
            self._set_status("No transcript entries to summarize")
            return
        transcript = self.summarizer.transcript_markdown(limit_chars=self.config.summary_max_transcript_chars)
        self._set_status("Generating Ollama meeting summary")
        self.dashboard.set_summary_running(True)
        task = SummaryTask(self.config, transcript)
        task.signals.report_ready.connect(self._on_summary_ready)
        task.signals.error.connect(self._set_status)
        task.signals.finished.connect(lambda: self.dashboard.set_summary_running(False))
        self.thread_pool.start(task)

    def _start_wav(self, wav_path: Path) -> None:
        task = WavTranscriptionTask(self.config, self.asr_engine, wav_path)
        task.signals.status.connect(self._set_status)
        task.signals.draft_ready.connect(self._on_draft)
        task.signals.error.connect(self._set_status)
        task.signals.finished.connect(self._on_wav_finished)
        self.thread_pool.start(task)

    @Slot(str, str)
    def _on_preview(self, speaker: str, text: str) -> None:
        self.overlay.update_preview(speaker, text)
        self.dashboard.set_preview(speaker, text)

    @Slot(object)
    def _on_partial_draft(self, draft: TranscriptDraft) -> None:
        self._apply_speaker_alias(draft)
        if draft.track_type in self._partial_busy_tracks:
            return
        self._partial_busy_tracks.add(draft.track_type)
        task = PartialTranscriptionTask(self.config, self.asr_engine, draft)
        task.signals.draft_ready.connect(self._on_partial_text)
        task.signals.error.connect(self._set_status)
        task.signals.finished.connect(lambda track=draft.track_type: self._partial_busy_tracks.discard(track))
        self.thread_pool.start(task)

    @Slot(object)
    def _on_partial_text(self, draft: TranscriptDraft) -> None:
        self._apply_speaker_alias(draft)
        preview = f"{draft.text.strip()} ..."
        self.overlay.update_preview(draft.speaker, preview)
        self.dashboard.set_preview(draft.speaker, preview)

    @Slot(object)
    def _on_draft(self, draft: TranscriptDraft) -> None:
        self._apply_speaker_alias(draft)
        if not draft.text.strip() and draft.audio_data is not None:
            self._start_audio_transcription(draft)
            return
        if not draft.text.strip():
            self._set_status("Empty sentence skipped")
            return

        self._enqueue_translation(draft)

    def _enqueue_translation(self, draft: TranscriptDraft) -> None:
        if self._should_merge_short_sentence(draft):
            self._merge_or_hold(draft)
            return

        if len(self._translation_queue) >= self.config.translation_queue_limit:
            dropped = self._translation_queue.popleft()
            self._set_status(f"Translation queue full; dropped oldest [{dropped.speaker}] sentence")
        self._translation_queue.append(draft)
        self.dashboard.set_queue_depth(len(self._translation_queue))
        self._drain_translation_queue()

    def _enqueue_translation_now(self, draft: TranscriptDraft) -> None:
        if len(self._translation_queue) >= self.config.translation_queue_limit:
            dropped = self._translation_queue.popleft()
            self._set_status(f"Translation queue full; dropped oldest [{dropped.speaker}] sentence")
        self._translation_queue.append(draft)
        self.dashboard.set_queue_depth(len(self._translation_queue))
        self._drain_translation_queue()

    def _drain_translation_queue(self) -> None:
        if self._translation_busy or not self._translation_queue:
            return
        draft = self._translation_queue.popleft()
        self.dashboard.set_queue_depth(len(self._translation_queue))
        self._translation_busy = True
        self._set_status(f"Translating [{draft.speaker}]")
        self.overlay.update_preview(draft.speaker, draft.text)
        task = TranslationTask(self.config, draft, self.summarizer.context_history())
        task.signals.translation_ready.connect(self._on_translation)
        task.signals.error.connect(self._set_status)
        task.signals.finished.connect(self._on_translation_finished)
        self.thread_pool.start(task)

    def _start_audio_transcription(self, draft: TranscriptDraft) -> None:
        task = AudioTranscriptionTask(self.config, self.asr_engine, draft)
        task.signals.status.connect(self._set_status)
        task.signals.draft_ready.connect(self._on_draft)
        task.signals.error.connect(self._set_status)
        self.thread_pool.start(task)

    @Slot(object, str)
    def _on_translation(self, draft: TranscriptDraft, translated: str) -> None:
        entry = TranscriptEntry(
            speaker=draft.speaker,
            original_text=draft.text,
            chinese_translation=translated,
            language_code=draft.language_code,
            track_type=draft.track_type,
            start_time=draft.start_time,
            end_time=draft.end_time,
            confidence=draft.confidence,
            captured_at=draft.captured_at,
            asr_started_at=draft.asr_started_at,
            asr_completed_at=draft.asr_completed_at,
            translation_started_at=draft.translation_started_at,
            translation_completed_at=draft.translation_completed_at,
        )
        self.summarizer.add_entry(entry)
        self.overlay.update_final(entry)
        self.dashboard.add_entry(entry)
        self.dashboard.set_latency(entry.latency_label)
        self._set_status(f"Ready | {entry.latency_label}" if entry.latency_label else "Ready")

    @Slot()
    def _on_translation_finished(self) -> None:
        self._translation_busy = False
        self._drain_translation_queue()

    @Slot(str)
    def _on_summary_ready(self, report: str) -> None:
        self._latest_ai_summary = report
        markdown = self.summarizer.build_markdown_report(ai_summary=report)
        self.dashboard.set_report(markdown)
        self._set_status("Ollama meeting summary ready")

    def _set_status(self, message: str) -> None:
        self._last_status = message
        self.dashboard.set_status(message)
        self.overlay.set_status(message) if message.lower().startswith("live loopback") else None

    @Slot(object)
    def _on_dashboard_settings_changed(self, settings: dict[str, object]) -> None:
        previous_model = self.config.asr_model_size
        apply_meeting_profile(self.config, str(settings["profile"]))
        apply_model_preset(self.config, str(settings["preset"]))
        self.config.translation_style = str(settings["style"])
        self.config.mic_device_index = settings["mic_device_index"]  # type: ignore[assignment]
        self.config.remote_device_index = settings["remote_device_index"]  # type: ignore[assignment]
        self.config.capture_mic_enabled = bool(settings["capture_mic_enabled"])
        self.config.capture_remote_enabled = bool(settings["capture_remote_enabled"])
        self.config.save_reports_enabled = bool(settings["save_reports_enabled"])
        self.config.privacy_mode = bool(settings["privacy_mode"])
        self.config.debug_audio_enabled = bool(settings["debug_audio_enabled"])
        self.config.speaker_aliases = dict(self.speaker_aliases)
        self.dashboard.sync_from_config()
        if previous_model != self.config.asr_model_size and not self._running_source:
            self.asr_engine = ASREngine(self.config)
        self._save_runtime_settings()
        note = "Settings updated and saved"
        if self._running_source:
            note += "; audio device/model changes apply after restart"
        self._set_status(note)

    @Slot(object)
    def _on_speaker_rename_requested(self, payload: dict[str, object]) -> None:
        old = str(payload.get("old", "")).strip()
        new = str(payload.get("new", "")).strip()
        if not old or not new or old == new:
            return
        for alias_key, alias_value in list(self.speaker_aliases.items()):
            if alias_value == old:
                self.speaker_aliases[alias_key] = new
        self.speaker_aliases[old] = new
        self.config.speaker_aliases = dict(self.speaker_aliases)
        for entry in self.summarizer.entries:
            if entry.speaker == old:
                entry.speaker = new
        self.dashboard.rename_speaker_entries(old, new)
        self._save_runtime_settings()
        self._set_status(f"Speaker alias saved: {old} -> {new}")

    @Slot()
    def _on_audio_finished(self) -> None:
        self._running_source = False
        self.dashboard.set_running_state(False)
        if "windows-only" in self._last_status.lower() or "not enabled" in self._last_status.lower():
            return
        self._set_status("Audio source finished")

    @Slot()
    def _on_wav_finished(self) -> None:
        self._running_source = False
        self.dashboard.set_running_state(False)
        self._set_status("WAV transcription finished")

    def _should_merge_short_sentence(self, draft: TranscriptDraft) -> bool:
        if not self.config.merge_short_sentences_enabled:
            return False
        if draft.track_type == "wav":
            return False
        if self._should_hold_for_german_clause(draft):
            return True
        return len(draft.text.strip()) <= self.config.merge_short_sentence_chars

    def _merge_or_hold(self, draft: TranscriptDraft) -> None:
        delay_ms = self._merge_delay_ms(draft)
        if self._merge_buffer is None:
            self._merge_buffer = draft
            self._merge_timer.start(delay_ms)
            return

        if self._merge_buffer.speaker == draft.speaker and self._merge_buffer.track_type == draft.track_type:
            self._merge_buffer.text = f"{self._merge_buffer.text.rstrip()} {draft.text.lstrip()}".strip()
            self._merge_buffer.end_time = max(self._merge_buffer.end_time, draft.end_time)
            self._merge_buffer.confidence = _average_confidence(
                self._merge_buffer.confidence,
                draft.confidence,
            )
            self._merge_timer.start(self._merge_delay_ms(self._merge_buffer))
            return

        self._flush_merge_buffer()
        self._merge_buffer = draft
        self._merge_timer.start(delay_ms)

    @Slot()
    def _flush_merge_buffer(self) -> None:
        if self._merge_buffer is None:
            return
        draft = self._merge_buffer
        self._merge_buffer = None
        self._enqueue_translation_now(draft)

    def _should_hold_for_german_clause(self, draft: TranscriptDraft) -> bool:
        if not self.config.german_clause_merge_enabled:
            return False
        if self.config.meeting_profile not in {"de", "de-en"}:
            return False
        if draft.language_code not in {"auto", "de"}:
            return False
        return is_german_clause_fragment(draft.text, self.config.german_clause_markers)

    def _merge_delay_ms(self, draft: TranscriptDraft) -> int:
        if self._should_hold_for_german_clause(draft):
            return self.config.german_clause_merge_ms
        return self.config.merge_short_sentence_ms

    def _apply_speaker_alias(self, draft: TranscriptDraft) -> None:
        if draft.speaker == "Me":
            return
        seen: set[str] = set()
        speaker = draft.speaker
        while speaker in self.speaker_aliases and speaker not in seen:
            seen.add(speaker)
            speaker = self.speaker_aliases[speaker]
        draft.speaker = speaker

    def _save_runtime_settings(self) -> None:
        try:
            save_runtime_settings(self.config)
        except Exception as exc:
            self._set_status(f"Settings save warning: {exc}")


def run_environment_check(
    config: AppConfig | None = None,
    profile: str | None = None,
    preset: str | None = None,
    style: str | None = None,
) -> int:
    cfg = config or load_config(profile=profile, preset=preset, translation_style=style)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ollama model: {cfg.ollama_model}")
    print(f"Meeting profile: {cfg.meeting_profile} ({cfg.language_profile_label})")
    print(f"Model preset: {cfg.model_preset} ({cfg.model_preset_label})")
    print(f"Translation style: {cfg.translation_style}")
    print(f"ASR model: {cfg.asr_model_size} / {cfg.asr_device} / {cfg.asr_compute_type}")
    allowed_languages = ",".join(cfg.asr_allowed_languages)
    force_language = cfg.asr_force_language or "auto"
    print(f"ASR languages: allowed={allowed_languages} / force={force_language} / fallback={cfg.asr_default_language}")
    partial = "on" if cfg.partial_subtitles_enabled else "off"
    print(f"VAD mode: {cfg.vad_mode} / partial subtitles: {partial}")
    print(f"Custom/profile terms: {len(cfg.profile_terms_text.split())} tokens")
    print(f"Saved settings: {cfg.settings_file}")
    print(f"Speaker aliases: {len(cfg.speaker_aliases)}")
    print(f"Privacy: save_reports={cfg.save_reports_enabled} / privacy_mode={cfg.privacy_mode}")
    _print_acceleration_hint(cfg)
    devices = sounddevice.query_devices()
    print(f"Audio devices: {len(devices)}")
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            print(f"  input[{index}]: {device['name']}")
    if cfg.is_windows:
        _print_windows_loopback_devices(cfg)
    if cfg.is_macos:
        _print_macos_remote_devices(cfg)

    async def check_ollama() -> str:
        return await LLMRefiner(cfg).healthcheck()

    print(f"Ollama health: {asyncio.run(check_ollama())}")
    return 0


def _print_windows_loopback_devices(config: AppConfig) -> None:
    try:
        import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"WASAPI loopback: unavailable ({exc})")
        return

    pa = pyaudio.PyAudio()
    try:
        loopbacks = _loopback_device_candidates(pa)
        print(f"WASAPI loopback devices: {len(loopbacks)}")
        for device in loopbacks:
            index = device.get("index", "?")
            name = device.get("name", "unknown")
            channels = device.get("maxInputChannels") or device.get("maxOutputChannels") or "?"
            sample_rate = device.get("defaultSampleRate", "?")
            print(f"  loopback[{index}]: {name} / {channels} ch / {sample_rate} Hz")
        try:
            selected = _select_wasapi_loopback_device(pa, pyaudio, config.loopback_device_index)
            print(f"Selected loopback: [{selected.get('index')}] {selected.get('name')}")
        except Exception as exc:
            print(f"Selected loopback: unavailable ({exc})")
    finally:
        pa.terminate()


def _print_macos_remote_devices(config: AppConfig) -> None:
    candidates = _remote_input_device_candidates(config)
    print(f"macOS remote input candidates: {len(candidates)}")
    for device in candidates:
        index = device.get("index", "?")
        name = device.get("name", "unknown")
        channels = device.get("max_input_channels", "?")
        sample_rate = device.get("default_samplerate", "?")
        print(f"  remote[{index}]: {name} / {channels} ch / {sample_rate} Hz")

    try:
        selected = _select_remote_input_device(config)
    except Exception as exc:
        print(f"Selected macOS remote input: unavailable ({exc})")
        return

    if selected is None:
        print("Selected macOS remote input: none")
    else:
        print(f"Selected macOS remote input: [{selected.get('index')}] {selected.get('name')}")


def _print_acceleration_hint(config: AppConfig) -> None:
    if config.is_windows:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            result = None
        if result and result.returncode == 0 and result.stdout.strip():
            print(f"Acceleration: NVIDIA detected ({result.stdout.strip().splitlines()[0]})")
            print("Acceleration hint: consider LMC_ASR_DEVICE=cuda and LMC_ASR_COMPUTE_TYPE=float16")
        else:
            print("Acceleration: no NVIDIA GPU detected by nvidia-smi")
        return

    if config.is_macos and platform.machine() == "arm64":
        print("Acceleration: Apple Silicon detected; current stable preset uses CPU int8")


def run_app(args: argparse.Namespace) -> int:
    config = load_config(profile=args.profile, preset=args.preset, translation_style=args.style)
    _apply_runtime_arg_overrides(config, args)
    app = QApplication(sys.argv)
    app.setApplicationName(config.app_name)
    controller = MeetingAppController(args, config)
    app.setProperty("meeting_controller", controller)
    controller.show()
    if args.autostart:
        QTimer.singleShot(250, controller.start)
    return app.exec()


def is_german_clause_fragment(text: str, markers: tuple[str, ...]) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    lowered = _normalise_german_text(stripped)
    tokens = [token.strip(".,;:!?()[]{}\"'") for token in lowered.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return False

    marker_set = {_normalise_german_text(marker) for marker in markers}
    if tokens[0] in marker_set:
        return True
    if tokens[-1] in marker_set:
        return True
    if stripped.endswith(","):
        return True
    if stripped[-1:] not in {".", "!", "?"} and any(token in marker_set for token in tokens[-8:]):
        return True
    return False


def _normalise_german_text(text: str) -> str:
    return (
        text.lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _average_confidence(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _apply_runtime_arg_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.privacy:
        config.privacy_mode = True
        config.save_reports_enabled = False
    if args.debug_audio:
        config.debug_audio_enabled = True
        config.ensure_directories()
    if args.no_mic_track:
        config.capture_mic_enabled = False
    if args.no_remote_track:
        config.capture_remote_enabled = False
