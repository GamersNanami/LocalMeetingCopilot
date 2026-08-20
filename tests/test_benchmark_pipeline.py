from __future__ import annotations

import json
import wave
from pathlib import Path

from scripts.benchmark_pipeline import (
    BenchmarkResult,
    audio_duration_seconds,
    build_variants,
    load_manifest,
    load_samples,
    parse_args,
    write_results,
)


def test_manifest_sample_paths_are_relative_to_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "de_sample",
                        "path": "audio/de_sample.wav",
                        "language": "de",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    samples = load_samples(manifest, manifest_path.parent)

    assert samples[0].path == tmp_path / "audio" / "de_sample.wav"
    assert samples[0].language == "de"


def test_build_variants_uses_manifest_matrix() -> None:
    args = parse_args(["--manifest", "benchmarks/manifest.example.json"])
    variants = build_variants(
        {
            "matrix": {
                "profiles": ["de"],
                "presets": ["fast", "balanced"],
                "styles": ["meeting"],
                "ollama_models": ["qwen2.5:3b-instruct"],
            }
        },
        args,
    )

    assert [variant.variant_id for variant in variants] == [
        "de_fast_meeting_qwen2.5-3b-instruct",
        "de_balanced_meeting_qwen2.5-3b-instruct",
    ]


def test_audio_duration_reads_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16_000)

    assert audio_duration_seconds(wav_path) == 1.0


def test_write_results_creates_json_csv_and_markdown(tmp_path: Path) -> None:
    result = BenchmarkResult(
        sample_id="de_sample",
        status="ok",
        profile="de",
        preset="fast",
        style="meeting",
        ollama_model="qwen2.5:3b-instruct",
        skip_llm=False,
        audio_path="benchmarks/audio/de_sample.wav",
        language="de",
        audio_seconds=2.0,
        asr_seconds=1.0,
        llm_seconds=0.5,
        total_seconds=1.5,
        total_realtime_factor=0.75,
    )

    write_results(tmp_path, {"name": "unit"}, [result])

    assert (tmp_path / "results.json").exists()
    assert "de_sample" in (tmp_path / "results.csv").read_text(encoding="utf-8")
    assert "Benchmark Report: unit" in (tmp_path / "report.md").read_text(encoding="utf-8")
