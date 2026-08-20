from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranslationCacheKey:
    text: str
    profile: str
    style: str
    model: str
    language_code: str
    glossary_hash: str

    @property
    def digest(self) -> str:
        payload = {
            "text": normalise_cache_text(self.text),
            "profile": self.profile,
            "style": self.style,
            "model": self.model,
            "language_code": self.language_code or "auto",
            "glossary_hash": self.glossary_hash,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class TranslationCacheEntry:
    key: str
    text: str
    translated_text: str
    profile: str
    style: str
    model: str
    language_code: str
    glossary_hash: str
    created_at: float
    updated_at: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> TranslationCacheEntry | None:
        try:
            key = str(payload["key"])
            translated_text = str(payload["translated_text"])
        except KeyError:
            return None
        if not key or not translated_text:
            return None
        created_at = _float_value(payload.get("created_at")) or time.time()
        updated_at = _float_value(payload.get("updated_at")) or created_at
        return cls(
            key=key,
            text=str(payload.get("text", "")),
            translated_text=translated_text,
            profile=str(payload.get("profile", "")),
            style=str(payload.get("style", "")),
            model=str(payload.get("model", "")),
            language_code=str(payload.get("language_code", "auto")),
            glossary_hash=str(payload.get("glossary_hash", "")),
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass(frozen=True, slots=True)
class TranslationCacheHit:
    translated_text: str
    source: str


class TranslationCache:
    def __init__(
        self,
        *,
        path: Path | None = None,
        enabled: bool = True,
        persist: bool = False,
        max_entries: int = 2000,
        ttl_seconds: float = 30 * 24 * 60 * 60,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.persist = persist and path is not None
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._now = now
        self._entries: dict[str, TranslationCacheEntry] = {}
        self._disk_keys: set[str] = set()
        if self.enabled and self.persist:
            self._load_from_disk()

    def get(self, key: TranslationCacheKey) -> TranslationCacheHit | None:
        if not self.enabled:
            return None
        digest = key.digest
        entry = self._entries.get(digest)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._entries.pop(digest, None)
            self._disk_keys.discard(digest)
            self._write_disk_snapshot()
            return None
        source = "disk" if digest in self._disk_keys else "memory"
        return TranslationCacheHit(translated_text=entry.translated_text, source=source)

    def put(self, key: TranslationCacheKey, translated_text: str) -> None:
        if not self.enabled:
            return
        text = translated_text.strip()
        if not text:
            return
        now = self._now()
        digest = key.digest
        previous = self._entries.get(digest)
        self._entries[digest] = TranslationCacheEntry(
            key=digest,
            text=key.text,
            translated_text=text,
            profile=key.profile,
            style=key.style,
            model=key.model,
            language_code=key.language_code,
            glossary_hash=key.glossary_hash,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._disk_keys.discard(digest)
        self._prune()
        self._write_disk_snapshot()

    def __len__(self) -> int:
        return len(self._entries)

    def _load_from_disk(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            entry = TranslationCacheEntry.from_json_dict(payload)
            if entry is None or self._is_expired(entry):
                continue
            self._entries[entry.key] = entry
            self._disk_keys.add(entry.key)
        self._prune()

    def _prune(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        ordered = sorted(self._entries.values(), key=lambda entry: entry.updated_at, reverse=True)
        keep = {entry.key for entry in ordered[: self.max_entries]}
        for key in list(self._entries):
            if key not in keep:
                self._entries.pop(key, None)
                self._disk_keys.discard(key)

    def _is_expired(self, entry: TranslationCacheEntry) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return self._now() - entry.updated_at > self.ttl_seconds

    def _write_disk_snapshot(self) -> None:
        if not self.persist or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(entry.to_json_dict(), ensure_ascii=False, sort_keys=True)
            for entry in sorted(self._entries.values(), key=lambda item: item.updated_at)
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._disk_keys = {entry.key for entry in self._entries.values()}


_CACHE_REGISTRY: dict[tuple[Path | None, bool, bool, int, float], TranslationCache] = {}


def get_translation_cache(
    *,
    path: Path | None,
    enabled: bool,
    persist: bool,
    max_entries: int,
    ttl_seconds: float,
) -> TranslationCache:
    key = (
        path,
        enabled,
        persist,
        max(1, int(max_entries)),
        max(0.0, float(ttl_seconds)),
    )
    cache = _CACHE_REGISTRY.get(key)
    if cache is None:
        cache = TranslationCache(
            path=path,
            enabled=enabled,
            persist=persist,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )
        _CACHE_REGISTRY[key] = cache
    return cache


def clear_translation_cache_registry() -> None:
    _CACHE_REGISTRY.clear()


def make_translation_cache_key(
    *,
    text: str,
    profile: str,
    style: str,
    model: str,
    language_code: str,
    glossary_prompt: str,
) -> TranslationCacheKey:
    return TranslationCacheKey(
        text=text,
        profile=profile,
        style=style,
        model=model,
        language_code=language_code or "auto",
        glossary_hash=glossary_hash(glossary_prompt),
    )


def glossary_hash(glossary_prompt: str) -> str:
    normalised = " ".join(glossary_prompt.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def normalise_cache_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = (
        lowered.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    lowered = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())


def _float_value(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
