import numpy as np

from config import AppConfig
from visual_tracker import (
    VisualSpeakerTracker,
    _clean_ocr_text,
    _extract_ocr_texts,
    _largest_colored_rect,
    _pick_speaker_name,
)


def test_clean_ocr_text_removes_meeting_ui_noise() -> None:
    assert _clean_ocr_text(" Zoom 12:30 | Yuanxiao Yu muted ") == "Yuanxiao Yu"


def test_pick_speaker_name_uses_confidence_threshold() -> None:
    speaker = _pick_speaker_name(
        [("Teams recording", 0.99), ("Anna Schmidt", 0.91)],
        min_confidence=0.5,
    )

    assert speaker == "Anna Schmidt"


def test_extract_ocr_texts_supports_rapidocr_tuple_shape() -> None:
    raw = (
        [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Kai Mueller", 0.87),
        ],
        0.12,
    )

    assert _extract_ocr_texts(raw) == [("Kai Mueller", 0.87)]


def test_largest_colored_rect_detects_zoom_green_border() -> None:
    image = np.zeros((240, 400, 3), dtype=np.uint8)
    green_bgr = np.array([35, 209, 96], dtype=np.uint8)
    image[70:76, 120:280] = green_bgr
    image[164:170, 120:280] = green_bgr
    image[70:170, 120:126] = green_bgr
    image[70:170, 274:280] = green_bgr

    rect = _largest_colored_rect(image)

    assert rect is not None
    x, y, width, height = rect
    assert 110 <= x <= 125
    assert 65 <= y <= 75
    assert width >= 150
    assert height >= 95


def test_visual_tracker_uses_cached_speaker_when_detection_fails() -> None:
    tracker = VisualSpeakerTracker(AppConfig(visual_speaker_cache_seconds=30))
    tracker._cache_speaker("Anna Schmidt")

    result = tracker.detect()

    assert result.speaker == "Anna Schmidt"
    assert result.reason == "cached speaker"
