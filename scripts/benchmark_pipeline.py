from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_engine import ASREngine  # noqa: E402
from config import AppConfig, load_config  # noqa: E402
from llm_refiner import LLMRefiner  # noqa: E402


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    sample_id: str
    path: Path
    language: str = "auto"
    speaker: str = "Remote Participant"
    reference_text: str = ""
    reference_translation: str = ""
    source: str = ""
    license: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, manifest_dir: Path) -> BenchmarkSample:
        sample_id = str(payload.get("id") or payload.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Benchmark sample is missing an id")
        raw_path = str(payload.get("path") or "").strip()
        if not raw_path:
            raise ValueError(f"Benchmark sample {sample_id!r} is missing a path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_dir / path
        tags = payload.get("tags", ())
        if isinstance(tags, str):
            tags = (tags,)
        return cls(
            sample_id=sample_id,
            path=path,
            language=str(payload.get("language", "auto")),
            speaker=str(payload.get("speaker", "Remote Participant")),
            reference_text=str(payload.get("reference_text", "")),
            reference_translation=str(payload.get("reference_translation", "")),
            source=str(payload.get("source", "")),
            license=str(payload.get("license", "")),
            notes=str(payload.get("notes", "")),
            tags=tuple(str(tag) for tag in tags),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkVariant:
    profile: str
    preset: str
    style: str
    ollama_model: str
    skip_llm: bool = False

    @property
    def variant_id(self) -> str:
        llm = "skip-llm" if self.skip_llm else self.ollama_model.replace(":", "-")
        return f"{self.profile}_{self.preset}_{self.style}_{llm}"


@dataclass(slots=True)
class BenchmarkResult:
    sample_id: str
    status: str
    profile: str
    preset: str
    style: str
    ollama_model: str
    skip_llm: bool
    audio_path: str
    language: str
    audio_seconds: float | None = None
    segment_count: int = 0
    asr_seconds: float | None = None
    llm_seconds: float | None = None
    total_seconds: float | None = None
    asr_realtime_factor: float | None = None
    total_realtime_factor: float | None = None
    transcript_chars: int = 0
    translation_chars: int = 0
    transcript: str = ""
    translation: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LocalMeetingCopilot ASR + LLM latency")
    parser.add_argument(
        "--manifest",
        default="benchmarks/manifest.example.json",
        help="Benchmark manifest JSON with sample paths and optional matrix settings",
    )
    parser.add_argument("--output-dir", default="logs/benchmarks")
    parser.add_argument("--profiles", nargs="+", help="Override profiles, e.g. de de-en")
    parser.add_argument("--presets", nargs="+", help="Override presets, e.g. fast balanced")
    parser.add_argument("--styles", nargs="+", help="Override translation styles")
    parser.add_argument("--ollama-models", nargs="+", help="Override Ollama models")
    parser.add_argument("--skip-llm", action="store_true", help="Only benchmark ASR")
    parser.add_argument("--limit", type=int, help="Limit samples per variant")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and print matrix")
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return an error when a listed audio file is missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    samples = load_samples(manifest, manifest_path.parent)
    variants = build_variants(manifest, args)
    if args.dry_run:
        print_dry_run(manifest, samples, variants)
        return 0

    run_dir = make_run_dir(Path(args.output_dir))
    results = run_benchmarks(
        samples=samples,
        variants=variants,
        limit=args.limit,
        fail_on_missing=args.fail_on_missing,
    )
    write_results(run_dir, manifest, results)
    print(f"Benchmark results written to: {run_dir}")
    if args.fail_on_missing and any(result.status == "missing" for result in results):
        return 2
    if any(result.status == "ok" for result in results):
        return 0
    return 1


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object")
    return payload


def load_samples(manifest: dict[str, Any], manifest_dir: Path) -> list[BenchmarkSample]:
    raw_samples = manifest.get("samples", [])
    if not isinstance(raw_samples, list):
        raise ValueError("Benchmark manifest field 'samples' must be a list")
    return [
        BenchmarkSample.from_dict(sample, manifest_dir=manifest_dir)
        for sample in raw_samples
        if isinstance(sample, dict)
    ]


def build_variants(manifest: dict[str, Any], args: argparse.Namespace) -> list[BenchmarkVariant]:
    matrix = manifest.get("matrix", {})
    if not isinstance(matrix, dict):
        matrix = {}
    profiles = args.profiles or _matrix_values(matrix, "profiles", ["de"])
    presets = args.presets or _matrix_values(matrix, "presets", ["fast"])
    styles = args.styles or _matrix_values(matrix, "styles", ["meeting"])
    skip_llm = bool(args.skip_llm)
    ollama_models = ["skip-llm"] if skip_llm else args.ollama_models or _matrix_values(
        matrix,
        "ollama_models",
        ["qwen2.5:3b-instruct"],
    )
    return [
        BenchmarkVariant(
            profile=profile,
            preset=preset,
            style=style,
            ollama_model=ollama_model,
            skip_llm=skip_llm,
        )
        for profile, preset, style, ollama_model in product(
            profiles,
            presets,
            styles,
            ollama_models,
        )
    ]


def run_benchmarks(
    *,
    samples: list[BenchmarkSample],
    variants: list[BenchmarkVariant],
    limit: int | None,
    fail_on_missing: bool,
) -> list[BenchmarkResult]:
    selected_samples = samples[:limit] if limit is not None else samples
    results: list[BenchmarkResult] = []
    for variant in variants:
        config = build_config(variant)
        asr_engine = ASREngine(config)
        refiner = None if variant.skip_llm else LLMRefiner(config)
        for sample in selected_samples:
            if not sample.path.exists():
                results.append(missing_result(sample, variant))
                if fail_on_missing:
                    continue
                continue
            results.append(run_sample(sample, variant, config, asr_engine, refiner))
    return results


def run_sample(
    sample: BenchmarkSample,
    variant: BenchmarkVariant,
    config: AppConfig,
    asr_engine: ASREngine,
    refiner: LLMRefiner | None,
) -> BenchmarkResult:
    result = BenchmarkResult(
        sample_id=sample.sample_id,
        status="ok",
        profile=variant.profile,
        preset=variant.preset,
        style=variant.style,
        ollama_model=variant.ollama_model,
        skip_llm=variant.skip_llm,
        audio_path=str(sample.path),
        language=sample.language,
        audio_seconds=audio_duration_seconds(sample.path),
    )
    total_started = time.perf_counter()
    try:
        asr_started = time.perf_counter()
        drafts = asr_engine.transcribe_file(sample.path, speaker=sample.speaker)
        result.asr_seconds = time.perf_counter() - asr_started
        result.segment_count = len(drafts)
        result.transcript = " ".join(draft.text for draft in drafts).strip()
        result.transcript_chars = len(result.transcript)

        if refiner is not None and drafts:
            llm_started = time.perf_counter()
            context_history: list[str] = []
            translations: list[str] = []
            for draft in drafts:
                translated = refiner.refine_and_translate_sync(
                    draft.text,
                    context_history=context_history,
                    language_code=draft.language_code or sample.language,
                )
                translations.append(translated)
                context_history.append(f"[{sample.speaker}] {draft.text}\n中文: {translated}")
            result.llm_seconds = time.perf_counter() - llm_started
            result.translation = " ".join(translations).strip()
            result.translation_chars = len(result.translation)
        result.total_seconds = time.perf_counter() - total_started
    except Exception as exc:
        result.status = "error"
        result.error = f"{exc.__class__.__name__}: {exc}"
        result.total_seconds = time.perf_counter() - total_started

    if result.audio_seconds and result.audio_seconds > 0:
        if result.asr_seconds is not None:
            result.asr_realtime_factor = result.asr_seconds / result.audio_seconds
        if result.total_seconds is not None:
            result.total_realtime_factor = result.total_seconds / result.audio_seconds
    return result


def build_config(variant: BenchmarkVariant) -> AppConfig:
    config = load_config(
        profile=variant.profile,
        preset=variant.preset,
        translation_style=variant.style,
    )
    if not variant.skip_llm:
        config.ollama_model = variant.ollama_model
    return config


def missing_result(sample: BenchmarkSample, variant: BenchmarkVariant) -> BenchmarkResult:
    return BenchmarkResult(
        sample_id=sample.sample_id,
        status="missing",
        profile=variant.profile,
        preset=variant.preset,
        style=variant.style,
        ollama_model=variant.ollama_model,
        skip_llm=variant.skip_llm,
        audio_path=str(sample.path),
        language=sample.language,
        error="Audio file not found",
    )


def audio_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
                return frames / frame_rate if frame_rate else None
        except Exception:
            return None
    try:
        import av  # type: ignore[import-not-found]

        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration * av.time_base)
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if audio_stream and audio_stream.duration and audio_stream.time_base:
                return float(audio_stream.duration * audio_stream.time_base)
    except Exception:
        return None
    return None


def write_results(run_dir: Path, manifest: dict[str, Any], results: list[BenchmarkResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "manifest_name": manifest.get("name", ""),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "results": [result.to_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(run_dir / "results.csv", results)
    (run_dir / "report.md").write_text(build_markdown_report(manifest, results), encoding="utf-8")


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    fields = [
        "sample_id",
        "status",
        "profile",
        "preset",
        "style",
        "ollama_model",
        "skip_llm",
        "language",
        "audio_seconds",
        "segment_count",
        "asr_seconds",
        "llm_seconds",
        "total_seconds",
        "asr_realtime_factor",
        "total_realtime_factor",
        "transcript_chars",
        "translation_chars",
        "audio_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.to_dict().get(field) for field in fields})


def build_markdown_report(manifest: dict[str, Any], results: list[BenchmarkResult]) -> str:
    ok_results = [result for result in results if result.status == "ok"]
    lines = [
        f"# Benchmark Report: {manifest.get('name', 'LocalMeetingCopilot')}",
        "",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total runs: {len(results)}",
        f"- Successful runs: {len(ok_results)}",
        "",
        "## Summary",
        "",
        "| Variant | Runs | Avg ASR s | Avg LLM s | Avg Total s | Avg Total RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_id in sorted({variant_key(result) for result in ok_results}):
        grouped = [result for result in ok_results if variant_key(result) == variant_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant_id,
                    str(len(grouped)),
                    _format_average(grouped, "asr_seconds"),
                    _format_average(grouped, "llm_seconds"),
                    _format_average(grouped, "total_seconds"),
                    _format_average(grouped, "total_realtime_factor"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Sample | Status | Variant | Audio s | ASR s | LLM s | Total s | Error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.sample_id,
                    result.status,
                    variant_key(result),
                    _format_number(result.audio_seconds),
                    _format_number(result.asr_seconds),
                    _format_number(result.llm_seconds),
                    _format_number(result.total_seconds),
                    result.error.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def print_dry_run(
    manifest: dict[str, Any],
    samples: list[BenchmarkSample],
    variants: list[BenchmarkVariant],
) -> None:
    print(f"Manifest: {manifest.get('name', '(unnamed)')}")
    print(f"Samples: {len(samples)}")
    for sample in samples:
        status = "ok" if sample.path.exists() else "missing"
        print(f"  - {sample.sample_id}: {sample.path} ({status})")
    print(f"Variants: {len(variants)}")
    for variant in variants:
        print(f"  - {variant.variant_id}")


def make_run_dir(output_dir: Path) -> Path:
    return output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")


def variant_key(result: BenchmarkResult) -> str:
    llm = "skip-llm" if result.skip_llm else result.ollama_model.replace(":", "-")
    return f"{result.profile}/{result.preset}/{result.style}/{llm}"


def _matrix_values(matrix: dict[str, Any], key: str, default: list[str]) -> list[str]:
    values = matrix.get(key, default)
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return default
    return [str(value) for value in values]


def _format_average(results: list[BenchmarkResult], field_name: str) -> str:
    values = [getattr(result, field_name) for result in results]
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return ""
    return _format_number(sum(numbers) / len(numbers))


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:0.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
