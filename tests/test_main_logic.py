from main import is_german_clause_fragment


def test_german_clause_fragment_detects_subordinate_start() -> None:
    assert is_german_clause_fragment("weil die Datenpipeline noch instabil ist", ("weil", "dass"))


def test_german_clause_fragment_detects_unfinished_connector() -> None:
    assert is_german_clause_fragment("Wir starten erst dann wenn", ("wenn",))


def test_german_clause_fragment_ignores_complete_plain_sentence() -> None:
    assert not is_german_clause_fragment("Wir starten morgen.", ("weil", "dass", "wenn"))
