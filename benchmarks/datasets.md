# Benchmark Audio Sources

This project does not commit third-party audio files to the repository. Put local benchmark clips in `benchmarks/audio/`, then update `benchmarks/manifest.example.json` or create your own manifest.

## Recommended First Sources

| Dataset | Languages | License | Why Use It | Notes |
| --- | --- | --- | --- | --- |
| [Mozilla Common Voice German](https://mozilladatacollective.com/datasets/cmqim3xpi00t6nr07k0myqtkr) | German | CC0-1.0 | Large German ASR corpus with diverse speakers. | Do not attempt to identify speakers. The German release is large, so extract only a few clips for benchmark work. |
| [Google FLEURS](https://huggingface.co/datasets/google/fleurs) | German, English, many more | CC-BY-4.0 | Controlled multilingual ASR/translation evaluation source. | Good for comparing `de`, `en`, and `de-en` channels because it has consistent metadata and references. |
| [LibriSpeech](https://www.openslr.org/12) | English | CC-BY-4.0 | Standard English ASR baseline. | Clean audiobook speech, not meeting-like. Use it to compare ASR settings, not to predict meeting accuracy. |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) | German, English, others | CC-BY-4.0 | Larger multilingual ASR corpus with German and English. | Very large downloads; prefer small dev/test subsets or a few extracted clips. |

## Suggested Local Layout

```text
benchmarks/
  audio/
    de_common_voice_sample.mp3
    de_fleurs_sample.wav
    en_librispeech_sample.flac
  manifest.local.json
```

Run a dry check:

```bash
python scripts/benchmark_pipeline.py --manifest benchmarks/manifest.local.json --dry-run
```

Run ASR-only first:

```bash
python scripts/benchmark_pipeline.py --manifest benchmarks/manifest.local.json --skip-llm
```

Then include Ollama translation:

```bash
python scripts/benchmark_pipeline.py --manifest benchmarks/manifest.local.json
```

Results are written under `logs/benchmarks/YYYYMMDD-HHMMSS/` as `results.json`, `results.csv`, and `report.md`.
