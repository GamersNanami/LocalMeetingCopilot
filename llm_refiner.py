from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import ollama

from config import AppConfig, load_config
from glossary import format_term_matches_for_prompt, load_glossary_terms, match_terms

TRANSLATOR_SYSTEM_PROMPT = """You are an expert bilingual meeting translator (German/English to Chinese).
Translate the input spoken sentence into accurate, fluent, contextual Simplified Chinese.
Special Rule: If the source language is German, handle Nebensatz / Rahmenkonstruktion (verb placed at the end) and reconstruct a natural sentence flow.
For German fragments with weil/dass/wenn/obwohl/damit/waehrend/bevor/nachdem/ob, infer the main-clause relationship from recent context before translating.
Preserve negation, modal verbs, deadlines, owners, dependencies, and business risk language exactly.
Return ONLY the final Chinese translation without explanations or conversational filler."""

TRANSLATOR_FAST_SYSTEM_PROMPT = """Translate German/English business meeting speech into concise Simplified Chinese.
For German, fix subordinate-clause order naturally before translating.
Preserve negation, owner, deadline, risk, and decision intent.
Return ONLY Chinese."""

SUMMARY_SYSTEM_PROMPT = """You are a meticulous bilingual meeting analyst.
Create a concise Simplified Chinese meeting report from the transcript.
Preserve named speakers and distinguish [Me] from other participants.
Return Markdown only, with exactly these sections:
# 1. 会议核心摘要与结论
# 2. 针对 [Me] 的专属任务与待办
# 3. 风险、未决问题与后续依赖"""


LOCAL_FILLER_TRANSLATIONS: dict[str, str] = {
    "ja": "嗯。",
    "ja genau": "对，没错。",
    "genau": "对。",
    "okay": "好的。",
    "ok": "好的。",
    "alles klar": "好的，明白。",
    "mhm": "嗯。",
    "hm": "嗯。",
    "äh": "嗯。",
    "ähm": "嗯。",
    "also": "那么。",
    "yes": "是的。",
    "yeah": "嗯，对。",
    "right": "对。",
    "sure": "好的。",
    "um": "嗯。",
    "uh": "嗯。",
    "嗯": "嗯。",
    "好": "好的。",
    "对": "对。",
}


class LLMRefiner:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.client = ollama.AsyncClient(
            host=self.config.ollama_host,
            timeout=self.config.ollama_timeout_seconds,
        )
        self.glossary_terms = (
            load_glossary_terms(
                profile=self.config.meeting_profile,
                profile_terms_dir=self.config.profile_terms_dir,
                custom_terms_file=self.config.custom_terms_file,
            )
            if self.config.structured_glossary_enabled
            else []
        )

    async def refine_and_translate(
        self,
        original_text: str,
        context_history: Sequence[str] | None = None,
        language_code: str = "auto",
    ) -> str:
        return await self._translate(
            original_text=original_text,
            context_history=context_history,
            language_code=language_code,
            stream=False,
        )

    async def refine_and_translate_stream(
        self,
        original_text: str,
        context_history: Sequence[str] | None = None,
        language_code: str = "auto",
        on_partial: Callable[[str], None] | None = None,
    ) -> str:
        return await self._translate(
            original_text=original_text,
            context_history=context_history,
            language_code=language_code,
            stream=True,
            on_partial=on_partial,
        )

    async def _translate(
        self,
        *,
        original_text: str,
        context_history: Sequence[str] | None,
        language_code: str,
        stream: bool,
        on_partial: Callable[[str], None] | None = None,
    ) -> str:
        text = original_text.strip()
        if not text:
            return ""

        local_translation = _local_filler_translation(text)
        if local_translation is not None:
            if on_partial:
                on_partial(local_translation)
            return local_translation

        context = "\n".join((context_history or [])[-self.config.context_window_size :])
        glossary_prompt = self.glossary_prompt_for_text(text)
        base_system_prompt = (
            TRANSLATOR_FAST_SYSTEM_PROMPT
            if self.config.model_preset == "fast"
            else TRANSLATOR_SYSTEM_PROMPT
        )
        system_prompt = (
            f"{base_system_prompt}\n\n"
            f"Meeting profile instruction: {self.config.translator_profile_instruction}"
        )
        if self.config.model_preset == "fast":
            prompt = build_translation_prompt(
                config=self.config,
                text=text,
                context=context,
                language_code=language_code,
                glossary_prompt=glossary_prompt,
                fast=True,
            )
        else:
            prompt = build_translation_prompt(
                config=self.config,
                text=text,
                context=context,
                language_code=language_code,
                glossary_prompt=glossary_prompt,
                fast=False,
            )
        options = {
            "temperature": 0.05 if self.config.model_preset == "fast" else 0.1,
            "num_predict": _translation_num_predict_for(text, self.config.translation_num_predict),
        }

        try:
            if stream:
                response_stream = await self.client.generate(
                    model=self.config.ollama_model,
                    system=system_prompt,
                    prompt=prompt,
                    stream=True,
                    options=options,
                )
                chunks: list[str] = []
                async for part in response_stream:
                    delta = _response_text(part)
                    if not delta:
                        continue
                    chunks.append(delta)
                    if on_partial:
                        on_partial("".join(chunks).strip())
                translated = "".join(chunks).strip()
                return translated or "（Ollama 未返回翻译，保留原文）"

            response = await self.client.generate(
                model=self.config.ollama_model,
                system=system_prompt,
                prompt=prompt,
                options=options,
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
                options={"temperature": 0.1, "num_predict": self.config.summary_num_predict},
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

    def refine_and_translate_stream_sync(
        self,
        original_text: str,
        context_history: Sequence[str] | None = None,
        language_code: str = "auto",
        on_partial: Callable[[str], None] | None = None,
    ) -> str:
        return asyncio.run(
            self.refine_and_translate_stream(
                original_text=original_text,
                context_history=context_history,
                language_code=language_code,
                on_partial=on_partial,
            )
        )

    def summarize_meeting_sync(self, transcript_markdown: str) -> str:
        return asyncio.run(self.summarize_meeting(transcript_markdown))

    def healthcheck_sync(self) -> str:
        return asyncio.run(self.healthcheck())

    def glossary_prompt_for_text(self, text: str) -> str:
        if not self.glossary_terms:
            return ""
        matches = match_terms(text, self.glossary_terms, limit=self.config.glossary_max_terms)
        return format_term_matches_for_prompt(matches)


def _response_text(response: object) -> str:
    if isinstance(response, dict):
        return str(response.get("response", ""))
    return str(getattr(response, "response", ""))


def _local_filler_translation(text: str) -> str | None:
    normalised = _normalise_filler_text(text)
    if not normalised:
        return None
    return LOCAL_FILLER_TRANSLATIONS.get(normalised)


def _normalise_filler_text(text: str) -> str:
    stripped = text.lower().strip()
    stripped = stripped.replace("…", " ")
    for char in ".,;:!?()[]{}\"'“”‘’":
        stripped = stripped.replace(char, " ")
    return " ".join(stripped.split())


def _translation_num_predict_for(text: str, base_limit: int) -> int:
    stripped = text.strip()
    if len(stripped) <= 30:
        return min(base_limit, 64)
    if len(stripped) <= 80:
        return min(base_limit, 96)
    return base_limit


def build_translation_prompt(
    *,
    config: AppConfig,
    text: str,
    context: str,
    language_code: str,
    glossary_prompt: str,
    fast: bool,
) -> str:
    glossary_section = ""
    if glossary_prompt:
        glossary_section = (
            "Relevant glossary terms. Use the Chinese target exactly when the term appears:\n"
            f"{glossary_prompt}\n\n"
        )
    if fast:
        return (
            f"Profile: {config.meeting_profile}; language: {language_code}\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"{glossary_section}"
            f"Text:\n{text}\n\n"
            "Chinese:"
        )
    return (
        f"Meeting profile: {config.meeting_profile} ({config.language_profile_label})\n"
        f"Detected language: {language_code}\n"
        f"Recent meeting context:\n{context or '(none)'}\n\n"
        f"{glossary_section}"
        f"Sentence to translate:\n{text}"
    )
