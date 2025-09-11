"""
Minimal MuMMIE core: simple DSPy agent with a small Cerebras adapter
and a direct REST helper for quick Q&A.
"""
from __future__ import annotations

from typing import Any

import dspy
import os
import requests


class SimpleQASignature(dspy.Signature):
    """Answer a user question concisely."""

    question: str = dspy.InputField(desc="User question")
    answer: str = dspy.OutputField(desc="Short, direct answer")


class MummieAgent(dspy.Module):
    """A minimal LLM agent powered by DSPy.

    Usage:
        >>> from mummie.core import configure_lm, MummieAgent
        >>> configure_lm(provider="ollama", model="llama3.1")
        >>> agent = MummieAgent(use_chain_of_thought=True)
        >>> agent.ask("What is 2 + 2?")
    """

    def __init__(self, use_chain_of_thought: bool = False) -> None:
        super().__init__()
        self.predictor = (
            dspy.ChainOfThought(SimpleQASignature)
            if use_chain_of_thought
            else dspy.Predict(SimpleQASignature)
        )

    def ask(self, question: str) -> str:
        prediction = self.predictor(question=question)
        return prediction.answer


def configure_lm(provider: str = "ollama", model: str = "llama3.1", **kwargs: Any) -> Any:
    """Configure DSPy's global language model.

    Supported providers: 'ollama', 'openai', 'mistral', 'lmstudio', 'cerebras'.
    Additional keyword arguments are passed to the provider constructor,
    e.g., timeout=60, api_key="...", base_url="...".
    """
    provider_to_class: dict[str, str] = {
        "ollama": "Ollama",
        "openai": "OpenAI",
        "mistral": "Mistral",
        "lmstudio": "LMStudio",
    }

    if provider == "cerebras":
        api_key = kwargs.pop("api_key", os.environ.get("CEREBRAS_API_KEY"))
        lm = CerebrasLM(model=model, api_key=api_key, **kwargs)
    elif provider in provider_to_class:
        class_name = provider_to_class[provider]
        lm_cls = getattr(dspy, class_name, None)
        if lm_cls is None:
            raise ImportError(
                f"dspy has no '{class_name}' backend. Install the appropriate extra or update dspy, "
                f"or use provider='cerebras'."
            )
        lm = lm_cls(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    dspy.configure(lm=lm)
    return lm


def answer(question: str, provider: str = "ollama", model: str = "llama3.1", use_cot: bool = True, **lm_kwargs: Any) -> str:
    """One-shot convenience helper to configure and query the agent in a single call."""
    configure_lm(provider=provider, model=model, **lm_kwargs)
    agent = MummieAgent(use_chain_of_thought=use_cot)
    return agent.ask(question)




class CerebrasLM(dspy.LM):
    """Working DSPy LM adapter for Cerebras Inference (OpenAI-compatible chat API)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.cerebras.ai",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 60,
        **_: Any,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CerebrasLM requires CEREBRAS_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs: Any):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if messages:
            messages_payload = messages
        elif prompt:
            messages_payload = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        payload = {
            "model": self.model,
            "messages": messages_payload,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        model = data.get("model", self.model)

        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class Choice:
            def __init__(self, content: str) -> None:
                self.message = Message(content)
                self.finish_reason = "stop"

        class Usage:
            def __init__(self, usage_dict: dict) -> None:
                self.prompt_tokens = usage_dict.get("prompt_tokens", 0)
                self.completion_tokens = usage_dict.get("completion_tokens", 0)
                self.total_tokens = usage_dict.get("total_tokens", 0)

            def __iter__(self):
                yield "prompt_tokens", self.prompt_tokens
                yield "completion_tokens", self.completion_tokens
                yield "total_tokens", self.total_tokens

        class Response:
            def __init__(self, content: str, usage_dict: dict, model_name: str) -> None:
                self.choices = [Choice(content)]
                self.usage = Usage(usage_dict)
                self.model = model_name

        return Response(content, usage, model)


class CerebrasAgent:
    """Simple direct Cerebras API agent (no DSPy dependency)."""

    def __init__(self, api_key: str | None = None, model: str = "qwen-3-32b", **kwargs: Any) -> None:
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("Need CEREBRAS_API_KEY")
        self.model = model
        self.base_url = "https://api.cerebras.ai"
        self.kwargs = kwargs
    def ask(self, question: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": self.kwargs.get("max_tokens", 256),
            "temperature": self.kwargs.get("temperature", 0.7),
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.ok:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"API Error: {resp.status_code} - {resp.text}")


def ask_direct(question: str, model: str = "qwen-3-32b", api_key: str | None = None, **kwargs: Any) -> str:
    """Minimal helper for direct Cerebras Q&A without configuring DSPy."""
    agent = CerebrasAgent(api_key=api_key, model=model, **kwargs)
    return agent.ask(question)



