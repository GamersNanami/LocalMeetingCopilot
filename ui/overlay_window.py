from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QContextMenuEvent, QMouseEvent
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

from config import AppConfig, load_config
from meeting_types import TranscriptEntry


class SubtitleOverlay(QWidget):
    open_dashboard_requested = Signal()
    close_requested = Signal()

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        self._drag_origin: QPoint | None = None

        self.setWindowTitle("LocalMeetingCopilot Overlay")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(520, 116)
        self.resize(self.config.overlay_width, self.config.overlay_height)
        self.setWindowOpacity(self.config.overlay_opacity)

        self.container = QWidget(self)
        self.container.setObjectName("overlayContainer")

        self.speaker_label = QLabel("[Ready]")
        self.speaker_label.setObjectName("speakerLabel")
        self.original_label = QLabel("Waiting for meeting audio...")
        self.original_label.setObjectName("originalLabel")
        self.original_label.setWordWrap(True)
        self.translation_label = QLabel("中文翻译会显示在这里")
        self.translation_label.setObjectName("translationLabel")
        self.translation_label.setWordWrap(True)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(8)
        layout.addWidget(self.speaker_label)
        layout.addWidget(self.original_label)
        layout.addWidget(self.translation_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)

        self.setStyleSheet(
            """
            QWidget#overlayContainer {
                background: rgba(18, 20, 24, 218);
                border: 1px solid rgba(255, 255, 255, 42);
                border-radius: 10px;
            }
            QLabel#speakerLabel {
                color: #8FD7FF;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#originalLabel {
                color: #F5F7FA;
                font-size: 20px;
                font-weight: 600;
            }
            QLabel#translationLabel {
                color: #FFD36A;
                font-size: 22px;
                font-weight: 700;
            }
            """
        )

    def update_preview(self, speaker: str, text: str) -> None:
        self.speaker_label.setText(f"[{speaker}]")
        self.original_label.setText(text or "...")

    def update_final(self, entry: TranscriptEntry) -> None:
        self.speaker_label.setText(f"[{entry.speaker}] {entry.timestamp}")
        self.original_label.setText(entry.original_text)
        self.translation_label.setText(entry.chinese_translation)

    def set_status(self, message: str) -> None:
        self.speaker_label.setText("[Status]")
        self.original_label.setText(message)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        open_dashboard = QAction("Open Dashboard", self)
        hide = QAction("Hide Overlay", self)
        close = QAction("Close", self)
        open_dashboard.triggered.connect(self.open_dashboard_requested.emit)
        hide.triggered.connect(self.hide)
        close.triggered.connect(self.close_requested.emit)
        menu.addAction(open_dashboard)
        menu.addAction(hide)
        menu.addSeparator()
        menu.addAction(close)
        menu.exec(event.globalPos())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_origin = None
        event.accept()
