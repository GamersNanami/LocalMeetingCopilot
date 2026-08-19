import numpy as np

import audio_engine
from audio_engine import (
    EnergyVADSegmenter,
    _remote_input_device_candidates,
    _resample_audio,
    _select_remote_input_device,
    _select_wasapi_loopback_device,
    create_vad_segmenter,
)
from config import AppConfig


def test_energy_vad_segments_voice_after_silence() -> None:
    config = AppConfig(
        audio_sample_rate=1000,
        audio_chunk_ms=100,
        vad_energy_threshold=0.05,
        vad_min_speech_ms=200,
        vad_silence_ms=200,
        vad_pre_roll_ms=100,
    )
    vad = EnergyVADSegmenter(config)
    silence = np.zeros(100, dtype=np.float32)
    voice = np.full(100, 0.2, dtype=np.float32)

    started_events = []
    completed = None
    for chunk in [silence, voice, voice, voice, silence, silence]:
        started, completed, _rms = vad.accept(chunk)
        started_events.append(started)
        if completed:
            break

    assert any(started_events)
    assert completed is not None
    assert completed.audio_data.size >= 300
    assert completed.start_time >= 0


def test_energy_vad_preview_audio_while_speaking() -> None:
    config = AppConfig(
        audio_sample_rate=1000,
        audio_chunk_ms=100,
        vad_energy_threshold=0.05,
        vad_min_speech_ms=200,
        vad_silence_ms=300,
        vad_pre_roll_ms=100,
    )
    vad = EnergyVADSegmenter(config)
    voice = np.full(100, 0.2, dtype=np.float32)

    vad.accept(voice)
    vad.accept(voice)
    preview = vad.preview_audio()

    assert preview is not None
    assert preview.audio_data.size >= 200
    assert preview.end_time > preview.start_time


def test_create_vad_segmenter_energy_mode() -> None:
    vad = create_vad_segmenter(AppConfig(vad_mode="energy"))

    assert isinstance(vad, EnergyVADSegmenter)
    assert vad.backend_name == "energy"


def test_resample_audio_converts_48k_to_16k() -> None:
    audio = np.ones(4800, dtype=np.float32)
    resampled = _resample_audio(audio, 48_000, 16_000)

    assert 1500 <= resampled.size <= 1700
    assert resampled.dtype == np.float32


def test_select_wasapi_loopback_prefers_default_output_match() -> None:
    class FakePyAudio:
        def get_host_api_info_by_type(self, _host_api: int) -> dict[str, int]:
            return {"defaultOutputDevice": 1}

        def get_device_info_by_index(self, index: int) -> dict[str, object]:
            devices = {
                1: {"index": 1, "name": "Speakers (Realtek Audio)", "isLoopbackDevice": False},
                3: {"index": 3, "name": "Headphones [Loopback]", "isLoopbackDevice": True},
                4: {"index": 4, "name": "Speakers (Realtek Audio) [Loopback]", "isLoopbackDevice": True},
            }
            return devices[index]

        def get_loopback_device_info_generator(self):
            yield self.get_device_info_by_index(3)
            yield self.get_device_info_by_index(4)

    class FakePyAudioModule:
        paWASAPI = 13

    selected = _select_wasapi_loopback_device(FakePyAudio(), FakePyAudioModule(), None)

    assert selected["index"] == 4


def test_select_remote_input_prefers_virtual_audio_device(monkeypatch) -> None:
    devices = [
        {"name": "MacBook Air Microphone", "max_input_channels": 1, "default_samplerate": 48_000},
        {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 48_000},
        {"name": "Built-in Output", "max_input_channels": 0, "default_samplerate": 48_000},
    ]

    def fake_query_devices(index=None):
        if index is None:
            return devices
        return devices[index]

    monkeypatch.setattr(audio_engine.sd, "query_devices", fake_query_devices)
    config = AppConfig()

    candidates = _remote_input_device_candidates(config)
    selected = _select_remote_input_device(config)

    assert [candidate["index"] for candidate in candidates] == [1]
    assert selected is not None
    assert selected["name"] == "BlackHole 2ch"


def test_select_remote_input_respects_explicit_index(monkeypatch) -> None:
    devices = [
        {"name": "MacBook Air Microphone", "max_input_channels": 1, "default_samplerate": 48_000},
        {"name": "Microsoft Teams Audio", "max_input_channels": 2, "default_samplerate": 48_000},
    ]

    def fake_query_devices(index=None):
        if index is None:
            return devices
        return devices[index]

    monkeypatch.setattr(audio_engine.sd, "query_devices", fake_query_devices)

    selected = _select_remote_input_device(AppConfig(remote_device_index=1))

    assert selected is not None
    assert selected["index"] == 1
