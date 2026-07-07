"""Minimal LLM client: uses Anthropic if ANTHROPIC_API_KEY is set,
otherwise OpenAI (matching the key this repo's app.py already uses).
Returns plain text; JSON parsing is the caller's problem."""

import json
import os
import re


class LLM:
    def __init__(self):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if anthropic_key:
            from anthropic import Anthropic

            self._anthropic = Anthropic(api_key=anthropic_key)
            self._openai = None
            self.model = os.getenv("AGENT_MODEL", "claude-sonnet-5")
        elif openai_key:
            from openai import OpenAI

            self._anthropic = None
            self._openai = OpenAI(api_key=openai_key)
            self.model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
        else:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment/.env "
                "— the agent needs a language model to think with."
            )

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if self._anthropic:
            resp = self._anthropic.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        resp = self._openai.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.8,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, tolerating code fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
