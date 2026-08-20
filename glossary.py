from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PRIORITY_WEIGHTS = {
    "high": 300,
    "medium": 200,
    "low": 100,
}


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    source: str
    variants: tuple[str, ...] = ()
    zh: str = ""
    category: str = "general"
    priority: str = "medium"
    profiles: tuple[str, ...] = ()
    origin: str = ""
    legacy: bool = False

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.source, *self.variants)


@dataclass(frozen=True, slots=True)
class TermMatch:
    term: GlossaryTerm
    matched_phrase: str
    score: float


def load_glossary_terms(
    *,
    profile: str,
    profile_terms_dir: Path,
    custom_terms_file: Path,
    include_legacy: bool = True,
) -> list[GlossaryTerm]:
    terms: list[GlossaryTerm] = []
    for path in structured_terms_files(profile_terms_dir, profile):
        terms.extend(load_structured_terms(path, profile=profile))
    if include_legacy:
        terms.extend(load_legacy_terms(profile, profile_terms_dir, custom_terms_file))
    return terms


def structured_terms_files(profile_terms_dir: Path, profile: str) -> tuple[Path, ...]:
    return (
        profile_terms_dir / "terms.yaml",
        profile_terms_dir / f"{profile}_terms.yaml",
    )


def load_structured_terms(path: Path, *, profile: str) -> list[GlossaryTerm]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_terms = payload.get("terms", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_terms, list):
        raise ValueError(f"Structured glossary must be a list or terms object: {path}")

    terms: list[GlossaryTerm] = []
    for raw_term in raw_terms:
        if not isinstance(raw_term, dict):
            continue
        term = _term_from_mapping(raw_term, origin=path.name)
        if term and _term_applies_to_profile(term, profile):
            terms.append(term)
    return terms


def load_legacy_terms(
    profile: str,
    profile_terms_dir: Path,
    custom_terms_file: Path,
) -> list[GlossaryTerm]:
    terms: list[GlossaryTerm] = []
    for path in (profile_terms_dir / f"{profile}_terms.txt", custom_terms_file):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            source = line.strip()
            if not source or source.startswith("#"):
                continue
            terms.append(
                GlossaryTerm(
                    source=source,
                    category="legacy",
                    priority="low",
                    origin=path.name,
                    legacy=True,
                )
            )
    return terms


def match_terms(
    text: str,
    terms: list[GlossaryTerm],
    *,
    limit: int = 8,
) -> list[TermMatch]:
    normalised_text = normalise_term_text(text)
    compact_text = compact_term_text(text)
    best_by_source: dict[str, TermMatch] = {}
    for term in terms:
        match = _best_match_for_term(term, normalised_text, compact_text)
        if match is None:
            continue
        key = normalise_term_text(term.source)
        previous = best_by_source.get(key)
        if previous is None or match.score > previous.score:
            best_by_source[key] = match
    matches = sorted(
        best_by_source.values(),
        key=lambda item: (
            item.score,
            len(normalise_term_text(item.term.source)),
            item.term.source.lower(),
        ),
        reverse=True,
    )
    return matches[: max(0, limit)]


def format_term_matches_for_prompt(matches: list[TermMatch]) -> str:
    lines: list[str] = []
    for match in matches:
        term = match.term
        target = f" => {term.zh}" if term.zh else ""
        variants = f"; variants: {', '.join(term.variants)}" if term.variants else ""
        lines.append(
            f"- {term.source}{target} "
            f"(category: {term.category}; priority: {term.priority}{variants})"
        )
    return "\n".join(lines)


def normalise_term_text(text: str) -> str:
    lowered = _normalise_german_letters(text.lower())
    lowered = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())


def compact_term_text(text: str) -> str:
    return normalise_term_text(text).replace(" ", "")


def _term_from_mapping(payload: dict[str, Any], *, origin: str) -> GlossaryTerm | None:
    source = str(payload.get("source", "")).strip()
    if not source:
        return None
    return GlossaryTerm(
        source=source,
        variants=_string_tuple(payload.get("variants", ())),
        zh=str(payload.get("zh") or payload.get("target_zh") or payload.get("chinese") or "").strip(),
        category=str(payload.get("category", "general")).strip() or "general",
        priority=_normalise_priority(str(payload.get("priority", "medium"))),
        profiles=_string_tuple(payload.get("profiles", ())),
        origin=origin,
    )


def _term_applies_to_profile(term: GlossaryTerm, profile: str) -> bool:
    if not term.profiles:
        return True
    return profile in term.profiles


def _best_match_for_term(
    term: GlossaryTerm,
    normalised_text: str,
    compact_text: str,
) -> TermMatch | None:
    best: TermMatch | None = None
    for phrase in term.phrases:
        phrase_norm = normalise_term_text(phrase)
        if not phrase_norm:
            continue
        matched = _normalised_phrase_matches(phrase_norm, normalised_text)
        compact_match = False
        if not matched and len(phrase_norm.replace(" ", "")) >= 8:
            compact_match = phrase_norm.replace(" ", "") in compact_text
        if not matched and not compact_match:
            continue
        score = _score_term_match(term, phrase_norm, source_match=phrase == term.source)
        candidate = TermMatch(term=term, matched_phrase=phrase, score=score)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _normalised_phrase_matches(phrase_norm: str, normalised_text: str) -> bool:
    return f" {phrase_norm} " in f" {normalised_text} "


def _score_term_match(term: GlossaryTerm, phrase_norm: str, *, source_match: bool) -> float:
    score = PRIORITY_WEIGHTS.get(term.priority, PRIORITY_WEIGHTS["medium"])
    score += min(len(phrase_norm), 80) / 10
    if source_match:
        score += 20
    if term.zh:
        score += 10
    if not term.legacy:
        score += 5
    return score


def _normalise_priority(priority: str) -> str:
    lowered = priority.lower().strip()
    return lowered if lowered in PRIORITY_WEIGHTS else "medium"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalise_german_letters(text: str) -> str:
    return (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
