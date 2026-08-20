from main import is_german_clause_fragment, should_skip_partial_transcription


def test_german_clause_fragment_detects_subordinate_start() -> None:
    assert is_german_clause_fragment("weil die Datenpipeline noch instabil ist", ("weil", "dass"))


def test_german_clause_fragment_detects_unfinished_connector() -> None:
    assert is_german_clause_fragment("Wir starten erst dann wenn", ("wenn",))


def test_german_clause_fragment_ignores_complete_plain_sentence() -> None:
    assert not is_german_clause_fragment("Wir starten morgen.", ("weil", "dass", "wenn"))


def test_partial_transcription_skips_when_final_asr_is_busy() -> None:
    assert should_skip_partial_transcription(
        asr_busy_count=1,
        partial_busy_tracks=set(),
        track_type="mic",
        skip_when_asr_busy=True,
    )


def test_partial_transcription_can_ignore_asr_busy_guard() -> None:
    assert not should_skip_partial_transcription(
        asr_busy_count=1,
        partial_busy_tracks=set(),
        track_type="mic",
        skip_when_asr_busy=False,
    )


def test_partial_transcription_skips_busy_track() -> None:
    assert should_skip_partial_transcription(
        asr_busy_count=0,
        partial_busy_tracks={"remote"},
        track_type="remote",
        skip_when_asr_busy=False,
    )
