from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from config import AppConfig, load_config
from meeting_types import TranscriptEntry


class MeetingSummarizer:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.entries: list[TranscriptEntry] = []

    def add_entry(self, entry: TranscriptEntry) -> None:
        self.entries.append(entry)

    def context_history(self, limit: int = 8) -> list[str]:
        recent = self.entries[-limit:]
        return [
            f"[{entry.speaker}] {entry.original_text} => {entry.chinese_translation}"
            for entry in recent
        ]

    def build_markdown_report(self, ai_summary: str | None = None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transcript = self.transcript_markdown()
        summary = ai_summary.strip() if ai_summary and ai_summary.strip() else self._build_local_summary()
        return (
            f"# LocalMeetingCopilot Meeting Report\n\n"
            f"Generated: {now}\n\n"
            f"{summary}\n\n"
            f"# 4. 完整中德/中英对照逐字记录\n\n"
            f"{transcript or '_No transcript entries yet._'}\n"
        )

    def transcript_markdown(self, limit_chars: int | None = None) -> str:
        transcript = "\n".join(entry.to_markdown() for entry in self.entries)
        if limit_chars is not None and len(transcript) > limit_chars:
            return transcript[-limit_chars:]
        return transcript

    def export_markdown(
        self,
        filepath: str | Path | None = None,
        ai_summary: str | None = None,
    ) -> Path:
        path = Path(filepath) if filepath else self._default_path("md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_markdown_report(ai_summary=ai_summary), encoding="utf-8")
        return path

    def export_json(self, filepath: str | Path | None = None) -> Path:
        path = Path(filepath) if filepath else self._default_path("json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _default_path(self, suffix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.log_dir / f"meeting_{stamp}.{suffix}"

    def _build_local_summary(self) -> str:
        me_entries = [entry for entry in self.entries if entry.speaker == "Me"]
        remote_entries = [entry for entry in self.entries if entry.speaker != "Me"]
        action_candidates = _find_action_candidates(self.entries)

        summary_lines = [
            "# 1. 会议核心摘要与结论",
            "",
            f"- 共记录 {len(self.entries)} 条发言。",
            f"- 远端发言 {len(remote_entries)} 条，[Me] 发言 {len(me_entries)} 条。",
        ]
        if remote_entries:
            summary_lines.append(f"- 最近的关键外部信息：{remote_entries[-1].chinese_translation}")
        if not self.entries:
            summary_lines.append("- 暂无会议内容。")

        action_lines = [
            "",
            "# 2. 针对 [Me] 的专属任务与待办",
            "",
        ]
        if action_candidates:
            action_lines.extend(f"- {item}" for item in action_candidates)
        else:
            action_lines.append("- 暂未检测到明确分配给 [Me] 的任务、问题或承诺。")

        return "\n".join(summary_lines + action_lines)


def _find_action_candidates(entries: Iterable[TranscriptEntry]) -> list[str]:
    markers = (
        "你",
        "我会",
        "我可以",
        "需要",
        "please",
        "could you",
        "can you",
        "i will",
        "ich kann",
        "ich werde",
        "bitte",
    )
    items: list[str] = []
    for entry in entries:
        combined = f"{entry.original_text} {entry.chinese_translation}".lower()
        if entry.speaker == "Me" or any(marker in combined for marker in markers):
            items.append(f"[{entry.timestamp}] [{entry.speaker}] {entry.chinese_translation}")
    return items[:12]
