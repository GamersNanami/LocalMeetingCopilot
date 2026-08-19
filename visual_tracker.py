from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from config import AppConfig, load_config

FALLBACK_SPEAKER = "Remote Participant"


@dataclass(slots=True)
class WindowCandidate:
    title: str
    left: int
    top: int
    width: int
    height: int


@dataclass(slots=True)
class SpeakerDetectionResult:
    speaker: str
    window_title: str | None = None
    active_rect: tuple[int, int, int, int] | None = None
    reason: str = "fallback"


class VisualSpeakerTracker:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._ocr: object | None = None
        self._cached_speaker: str | None = None
        self._cached_at = 0.0

    def detect_active_speaker(self) -> str:
        return self.detect().speaker

    def detect(self) -> SpeakerDetectionResult:
        if not self.config.visual_scan_enabled:
            return SpeakerDetectionResult(FALLBACK_SPEAKER, reason="visual scan disabled")

        windows = self._find_meeting_windows()
        if not windows:
            cached = self._cached_result()
            if cached:
                return cached
            return SpeakerDetectionResult(FALLBACK_SPEAKER, reason="no Teams/Zoom window")

        for window in windows:
            image = self._grab_window(window)
            if image is None:
                continue

            active_rect = _largest_colored_rect(image)
            if active_rect is None:
                continue

            name_region = _crop_name_region(image, active_rect)
            speaker = _pick_speaker_name(
                self._ocr_texts(name_region),
                min_confidence=self.config.visual_ocr_min_confidence,
            )
            if speaker:
                self._cache_speaker(speaker)
                return SpeakerDetectionResult(
                    speaker=speaker,
                    window_title=window.title,
                    active_rect=active_rect,
                    reason="ocr",
                )

        cached = self._cached_result()
        if cached:
            return cached
        return SpeakerDetectionResult(FALLBACK_SPEAKER, reason="active speaker OCR not found")

    def _cache_speaker(self, speaker: str) -> None:
        if speaker and speaker != FALLBACK_SPEAKER:
            self._cached_speaker = speaker
            self._cached_at = time.monotonic()

    def _cached_result(self) -> SpeakerDetectionResult | None:
        if not self._cached_speaker:
            return None
        age = time.monotonic() - self._cached_at
        if age > self.config.visual_speaker_cache_seconds:
            return None
        return SpeakerDetectionResult(self._cached_speaker, reason="cached speaker")

    def _find_meeting_windows(self) -> list[WindowCandidate]:
        try:
            import pygetwindow as gw  # type: ignore[import-not-found]
        except Exception:
            return []

        candidates: list[WindowCandidate] = []
        keywords = tuple(keyword.lower() for keyword in self.config.visual_window_keywords)
        for window in gw.getAllWindows():
            title = str(getattr(window, "title", "") or "")
            if not title or not any(keyword in title.lower() for keyword in keywords):
                continue
            if _window_is_minimized(window):
                continue

            left = int(getattr(window, "left", 0) or 0)
            top = int(getattr(window, "top", 0) or 0)
            width = int(getattr(window, "width", 0) or 0)
            height = int(getattr(window, "height", 0) or 0)
            if width < 240 or height < 180:
                continue
            candidates.append(WindowCandidate(title, left, top, width, height))

        return sorted(candidates, key=lambda item: item.width * item.height, reverse=True)

    def _grab_window(self, window: WindowCandidate) -> np.ndarray | None:
        try:
            import mss
        except Exception:
            return None

        monitor = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        try:
            with mss.mss() as capture:
                return np.asarray(capture.grab(monitor), dtype=np.uint8)
        except Exception:
            return None

    def _ocr_texts(self, image: np.ndarray) -> list[tuple[str, float | None]]:
        try:
            ocr = self._load_ocr()
            raw = ocr(image)
        except Exception:
            return []
        return _extract_ocr_texts(raw)

    def _load_ocr(self) -> object:
        if self._ocr is not None:
            return self._ocr

        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        except Exception:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]

        self._ocr = RapidOCR()
        return self._ocr


def _window_is_minimized(window: object) -> bool:
    value = getattr(window, "isMinimized", False)
    if callable(value):
        try:
            return bool(value())
        except Exception:
            return False
    return bool(value)


def _largest_colored_rect(image: np.ndarray) -> tuple[int, int, int, int] | None:
    if image.size == 0:
        return None

    try:
        import cv2
    except Exception:
        return None

    bgr = _as_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([35, 70, 70]), np.array([90, 255, 255])),
        cv2.inRange(hsv, np.array([90, 70, 70]), np.array([125, 255, 255])),
        cv2.inRange(hsv, np.array([125, 40, 60]), np.array([165, 255, 255])),
    ]
    mask = masks[0] | masks[1] | masks[2]
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[-2]

    height, width = bgr.shape[:2]
    min_area = max(2_500, int(width * height * 0.008))
    best_rect: tuple[int, int, int, int] | None = None
    best_score = 0
    for contour in contours:
        x, y, rect_width, rect_height = cv2.boundingRect(contour)
        area = rect_width * rect_height
        if area < min_area or rect_width < 80 or rect_height < 50:
            continue
        aspect = rect_width / max(rect_height, 1)
        if aspect < 0.65 or aspect > 3.5:
            continue
        if rect_width > width * 0.96 and rect_height > height * 0.96:
            continue

        score = area
        if score > best_score:
            best_score = score
            best_rect = (int(x), int(y), int(rect_width), int(rect_height))

    return best_rect


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] >= 3:
        return np.ascontiguousarray(image[:, :, :3])
    return np.repeat(image[:, :, :1], 3, axis=2)


def _crop_name_region(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    image_height, image_width = image.shape[:2]
    pad = max(4, int(min(width, height) * 0.04))
    x1 = max(0, x + pad)
    x2 = min(image_width, x + width - pad)
    y1 = min(image_height, y + int(height * 0.62))
    y2 = min(image_height, y + height - pad)
    if y2 - y1 < 18:
        y1 = max(0, y + height - 52)
        y2 = min(image_height, y + height - pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image[max(0, y) : min(image_height, y + height), max(0, x) : min(image_width, x + width)]
    return crop


def _extract_ocr_texts(raw: object) -> list[tuple[str, float | None]]:
    if raw is None:
        return []

    if hasattr(raw, "txts"):
        txts = list(raw.txts or [])
        scores = list(getattr(raw, "scores", []) or [])
        return [
            (str(text), _score_or_none(scores[index]) if index < len(scores) else None)
            for index, text in enumerate(txts)
        ]

    result = raw
    if isinstance(raw, tuple) and raw:
        result = raw[0]

    if not isinstance(result, list):
        return []

    texts: list[tuple[str, float | None]] = []
    for item in result:
        text, confidence = _parse_ocr_item(item)
        if text:
            texts.append((text, confidence))
    return texts


def _parse_ocr_item(item: object) -> tuple[str, float | None]:
    if isinstance(item, str):
        return item, None

    if not isinstance(item, (list, tuple)):
        return "", None

    if len(item) >= 3 and isinstance(item[1], str):
        return item[1], _score_or_none(item[2])
    if len(item) >= 2 and isinstance(item[0], str):
        return item[0], _score_or_none(item[1])

    for child in item:
        text, confidence = _parse_ocr_item(child)
        if text:
            return text, confidence
    return "", None


def _score_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        score = float(value)
        return score if 0.0 <= score <= 1.0 else None
    return None


def _pick_speaker_name(
    ocr_items: list[tuple[str, float | None]],
    *,
    min_confidence: float,
) -> str:
    for raw_text, confidence in ocr_items:
        if confidence is not None and confidence < min_confidence:
            continue
        candidate = _clean_ocr_text(raw_text)
        if _looks_like_name(candidate):
            return candidate
    return ""


def _clean_ocr_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " "))
    cleaned = re.sub(r"(?i)\b(microsoft teams|teams|zoom|meeting|recording)\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(muted|unmuted|speaking|speaker|participants?)\b", " ", cleaned)
    cleaned = re.sub(r"(?i)\byou are sharing\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", " ", cleaned)
    cleaned = re.sub(r"[|\\/\[\]{}<>_*#=]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-_")
    return cleaned[:64].strip()


def _looks_like_name(text: str) -> bool:
    if len(text) < 2:
        return False
    lower = text.lower()
    noise = (
        "meeting",
        "recording",
        "sharing",
        "mute",
        "camera",
        "chat",
        "screen",
        "participant",
    )
    if any(word in lower for word in noise):
        return False
    if not any(character.isalpha() for character in text):
        return False
    if sum(character.isdigit() for character in text) > 2:
        return False
    return True
