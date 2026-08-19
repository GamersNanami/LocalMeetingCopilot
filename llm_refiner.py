from __future__ import annotations

import asyncio
from collections.abc import Sequence

import ollama

from config import AppConfig, load_config

TRANSLATOR_SYSTEM_PROMPT = """You are an expert bilingual meeting translator (German/English to Chinese).
Translate the input spoken sentence into accurate, fluent, contextual Simplified Chinese.
Special Rule: If the source language is German, handle Nebensatz / Rahmenkonstruktion (verb placed at the end) and reconstruct a natural sentence flow.
For German fragments with weil/dass/wenn/obwohl/damit/waehrend/bevor/nachdem/ob, infer the main-clause relationship from recent context before translating.
Preserve negation, modal verbs, deadlines, owners, dependencies, and business risk language exactly.
Return ONLY the final Chinese translation without explanations or conversational filler."""

SUMMARY_SYSTEM_PROMPT = """You are a meticulous bilingual meeting analyst.
Create a concise Simplified Chinese meeting report from the transcript.
Preserve named speakers and distinguish [Me] from other participants.
Return Markdown only, with exactly these sections:
# 1. 会议核心摘要与结论
# 2. 针对 [Me] 的专属任务与待办
# 3. 风险、未决问题与后续依赖"""


class LLMRefiner:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.client = ollama.AsyncClient(host=self.config.ollama_host)

    async def refine_and_translate(
        self,
        original_text: str,
        context_history: Sequence[str] | None = None,
        language_code: str = "auto",
    ) -> str:
        text = original_text.strip()
        if not text:
            return ""

        context = "\n".join((context_history or [])[-self.config.context_window_size :])
        system_prompt = (
            f"{TRANSLATOR_SYSTEM_PROMPT}\n\n"
            f"Meeting profile instruction: {self.config.translator_profile_instruction}"
        )
        prompt = (
            f"Meeting profile: {self.config.meeting_profile} ({self.config.language_profile_label})\n"
            f"Detected language: {language_code}\n"
            f"Recent meeting context:\n{context or '(none)'}\n\n"
            f"Sentence to translate:\n{text}"
        )

        try:
            response = await self.client.generate(
                model=self.config.ollama_model,
                system=system_prompt,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 256},
            )
        except Exception as exc:
            return f"（Ollama 暂不可用，保留原文：{exc.__class__.__name__}）"

        translated = _response_text(response).strip()
        return translated or "（Ollama 未返回翻译，保留原文）"

    async def healthcheck(self) -> str:
        try:
            response = await self.client.generate(
                model=self.config.ollama_model,
                prompt="只回答两个字：正常",
                options={"temperature": 0, "num_predict": 4},
            )
        except Exception as exc:
            return f"offline: {exc.__class__.__name__}"
        return _response_text(response).strip() or "online"

    async def summarize_meeting(self, transcript_markdown: str) -> str:
        text = transcript_markdown.strip()
        if not text:
            return "暂无可总结的会议记录。"

        prompt = (
            "请基于以下逐字记录生成会议总结。重点识别直接分配给 [Me] 的任务、"
            "别人向 [Me] 提出的问题、[Me] 作出的承诺、技术风险和外部依赖。\n\n"
            f"{text}"
        )
        try:
            response = await self.client.generate(
                model=self.config.ollama_model,
                system=SUMMARY_SYSTEM_PROMPT,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 900},
            )
        except Exception as exc:
            return f"（Ollama 暂不可用，已使用本地规则摘要。错误：{exc.__class__.__name__}）"

        return _response_text(response).strip() or "（Ollama 未返回总结。）"

    def refine_and_translate_sync(
        self,
        original_text: str,
        context_history: Sequence[str] | None = None,
        language_code: str = "auto",
    ) -> str:
        return asyncio.run(
            self.refine_and_translate(
                original_text=original_text,
                context_history=context_history,
                language_code=language_code,
            )
        )

    def summarize_meeting_sync(self, transcript_markdown: str) -> str:
        return asyncio.run(self.summarize_meeting(transcript_markdown))


def _response_text(response: object) -> str:
    if isinstance(response, dict):
        return str(response.get("response", ""))
    return str(getattr(response, "response", ""))
