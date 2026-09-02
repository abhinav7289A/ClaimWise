"""The swappable generator slot: `generate(prompt) -> GenerationResult`.

Every LLM call in ClaimWise goes through this module. Nothing else in the
codebase may talk to a provider directly, because the whole Phase 4 benchmark
depends on being able to swap the generator without touching anything upstream:
if the retrieval pipeline is identical across generators, a difference in
answer quality is attributable to the generator alone.

**Why one class covers three providers.** NVIDIA NIM, OpenRouter and Hugging
Face Inference Providers all expose the OpenAI chat-completions protocol. They
differ only in base URL, API key and model name — so "provider" is configuration,
not a code path. Phase 5 adds a fourth entry pointing at the fine-tuned Qwen
Space; the interface does not change.

**Why temperature defaults to 0.** Evaluation has to be reproducible. A judged
metric that drifts because of sampling noise cannot be attributed to a pipeline
change, which would make the entire Phase 2 delta table meaningless.

**Why there is a client-side throttle.** NIM allows roughly 40 requests/minute.
A full RAGAS run is ~1,000 calls, so an unthrottled loop hits the limit within
seconds. Staying under it turns a rate-limit error into a short wait instead of
a failed hour-long eval run.

Usage:
    python -m common.generator --help
    python -m common.generator --prompt "Say OK if you can hear me."
    python -m common.generator --provider openrouter --prompt "Say OK."
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.generator")

# Errors worth retrying: transient network faults, provider rate limits, and
# 5xx. A 400 or 401 is a bug or a bad key and retrying only wastes quota.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GenerationResult:
    """One completion plus the accounting needed for metrics.

    Attributes:
        text: The generated answer.
        provider: Which configured provider served it, e.g. "nim".
        model: The exact model id used.
        prompt_tokens: Input tokens billed, as reported by the provider.
        completion_tokens: Output tokens generated.
        latency_seconds: Wall-clock time for the call.
    """

    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    @property
    def total_tokens(self) -> int:
        """Total tokens for this call — the "tokens per query" metric."""
        return self.prompt_tokens + self.completion_tokens


class Generator(Protocol):
    """The interface every generator must satisfy.

    Phase 5 swaps a fine-tuned Qwen endpoint into this slot. As long as it
    implements `generate`, no caller changes.
    """

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        """Produce a completion for `prompt`."""
        ...


class RateLimiter:
    """Sliding-window throttle that blocks until a request is allowed.

    Simpler and more predictable than reacting to 429s: we stay under the
    provider's limit rather than discovering it mid-run.
    """

    def __init__(self, requests_per_minute: int) -> None:
        """Initialise the limiter.

        Args:
            requests_per_minute: Maximum requests allowed in any 60s window.
                Zero or negative disables throttling.
        """
        self.requests_per_minute = requests_per_minute
        self._timestamps: deque[float] = deque()
        # Guards the check-then-append below. `agent_eval --workers N` shares one
        # generator across threads, and without this the length check and the
        # append can interleave so N threads all decide there is room for one
        # more — which turns a client-side throttle into a source of 429s.
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until issuing a request would not exceed the limit.

        Written as a loop rather than the obvious recursive call so the lock can
        be released while sleeping: a thread waiting out the window must not hold
        it, or the throttle serialises every worker behind the slowest one.
        """
        if self.requests_per_minute <= 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return

                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.05

            LOGGER.debug("Rate limit reached; sleeping %.2fs", sleep_for)
            time.sleep(max(sleep_for, 0.0))


class OpenAICompatGenerator:
    """A generator backed by any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_seconds: int = 60,
        max_retries: int = 4,
        requests_per_minute: int = 30,
    ) -> None:
        """Configure the client.

        Args:
            provider: Name of the configured provider, for logging and metrics.
            base_url: Provider's OpenAI-compatible endpoint.
            api_key: Secret, read from the environment by `build_generator`.
            model: Model id as the provider names it.
            temperature: Sampling temperature; 0.0 for reproducible evals.
            max_tokens: Cap on generated tokens.
            timeout_seconds: Per-request timeout.
            max_retries: Retry attempts for transient failures.
            requests_per_minute: Client-side throttle.
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._limiter = RateLimiter(requests_per_minute)
        # max_retries=0: we own the retry loop so backoff is visible and
        # rate-limit waits are logged rather than hidden inside the SDK.
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=0
        )

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        """Produce a completion, retrying transient failures with backoff.

        Args:
            prompt: The user message — for RAG this is the assembled prompt of
                question plus retrieved chunks.
            system: Optional system message carrying the grounding contract.

        Returns:
            The completion and its token accounting.

        Raises:
            RuntimeError: If every retry attempt failed. Raised loudly rather
                than returning an empty string, which would silently corrupt a
                metric downstream.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except (RateLimitError, APIConnectionError) as error:
                last_error = error
                self._sleep_before_retry(attempt, str(error))
                continue
            except APIStatusError as error:
                last_error = error
                if error.status_code not in RETRYABLE_STATUS:
                    raise RuntimeError(
                        f"{self.provider}/{self.model} returned "
                        f"{error.status_code}: {error.message}"
                    ) from error
                self._sleep_before_retry(attempt, f"HTTP {error.status_code}")
                continue

            latency = time.perf_counter() - started
            usage = response.usage
            return GenerationResult(
                text=(response.choices[0].message.content or "").strip(),
                provider=self.provider,
                model=self.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_seconds=round(latency, 3),
            )

        raise RuntimeError(
            f"{self.provider}/{self.model} failed after {self.max_retries + 1} "
            f"attempts. Last error: {last_error}"
        )

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        """Wait with exponential backoff before the next attempt.

        Args:
            attempt: Zero-based attempt number that just failed.
            reason: Short description for the log.
        """
        delay = min(2.0**attempt, 30.0)
        LOGGER.warning(
            "%s attempt %d failed (%s); retrying in %.1fs",
            self.provider,
            attempt + 1,
            reason,
            delay,
        )
        time.sleep(delay)


def build_generator(
    config: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
) -> OpenAICompatGenerator:
    """Construct the configured generator, reading its API key from the env.

    Args:
        config: Parsed `config.yaml`.
        provider: Override `generator.provider`.
        model: Override the provider's configured model.
        timeout_seconds: Override `generator.timeout_seconds`. The default 60 is
            sized for RAG prompts of ~1,500 tokens. Prefill time scales with
            input length, so a workload that sends a 124K-token prompt needs a
            different figure entirely — 60 seconds there is a timeout by
            construction, not a transient failure.
        max_retries: Override `generator.max_retries`. Worth lowering for very
            large prompts: the default backoff assumes retries are cheap, but
            re-sending 124K tokens pays the same prefill again, so three retries
            turn one slow request into four.

    Returns:
        A ready-to-use generator.

    Raises:
        ValueError: If the provider is not configured.
        RuntimeError: If the provider's API key is missing from the
            environment. Failing here is far cheaper than failing 400 calls
            into an eval run.
    """
    load_dotenv()

    provider_name = provider or cfg_get(config, "generator.provider", "nim")
    provider_config = cfg_get(config, f"generator.providers.{provider_name}")
    if not provider_config:
        available = sorted(cfg_get(config, "generator.providers", {}))
        raise ValueError(
            f"Provider {provider_name!r} is not configured. Available: {available}"
        )

    # A provider may nominate an env var that overrides its base_url. Added for
    # the Modal endpoint, whose URL differs between `modal serve` (ephemeral,
    # `-dev` suffixed) and `modal deploy` (persistent) and would otherwise need a
    # config edit to switch between — a config edit that is easy to commit by
    # accident and points the whole system at a dev endpoint that no longer
    # exists. Providers without `base_url_env` are unaffected.
    base_url = provider_config["base_url"]
    base_url_env = provider_config.get("base_url_env")
    if base_url_env and os.getenv(base_url_env):
        base_url = os.environ[base_url_env]
        LOGGER.info("%s base_url overridden by %s", provider_name, base_url_env)

    # Modal prints two URLs and only one of them is callable: the dashboard
    # (`modal.com/apps/<workspace>/...`) and the service (`*.modal.run`). Pasting
    # the dashboard one yields `405 Method Not Allowed` from an HTML page, an
    # error that says nothing about which URL is wrong and sends you looking at
    # the server. Caught here, with the fix in the message, because the failure
    # is otherwise indistinguishable from a broken endpoint.
    if "modal.com/apps" in base_url:
        raise RuntimeError(
            f"{base_url_env or 'base_url'} points at the Modal DASHBOARD, not the "
            f"service:\n  {base_url}\nThat page is HTML and returns 405 to a POST. "
            "Use the URL ending in `.modal.run`, printed by `modal deploy` and "
            "listed by `modal app list`, with `/v1` appended."
        )

    key_env = provider_config["api_key_env"]
    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {key_env} is not set. Copy .env.example to "
            f".env and add your {provider_name} key."
        )

    return OpenAICompatGenerator(
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        model=model or provider_config["model"],
        temperature=cfg_get(config, "generator.temperature", 0.0),
        max_tokens=cfg_get(config, "generator.max_tokens", 1024),
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else cfg_get(config, "generator.timeout_seconds", 60)
        ),
        max_retries=(
            max_retries if max_retries is not None else cfg_get(config, "generator.max_retries", 4)
        ),
        requests_per_minute=cfg_get(config, "generator.requests_per_minute", 30),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the connectivity-check CLI.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m common.generator",
        description="Verify a provider is reachable and a key works.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default=None, help="Override generator.provider.")
    parser.add_argument("--model", default=None, help="Override the provider's model.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt to send as a connectivity check.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Send one prompt and report the result, to prove a provider works.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if the call failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    generator = build_generator(config, provider=args.provider, model=args.model)

    print(f"provider : {generator.provider}")
    print(f"model    : {generator.model}")
    print(f"prompt   : {args.prompt!r}\n")

    try:
        result = generator.generate(args.prompt)
    except RuntimeError as error:
        LOGGER.error("%s", error)
        return 1

    print("=== GENERATOR CHECK ===")
    print(f"response : {result.text!r}")
    print(f"tokens   : {result.prompt_tokens} in / {result.completion_tokens} out")
    print(f"latency  : {result.latency_seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
