"""Serve the fine-tuned Qwen on Modal behind an OpenAI-compatible endpoint.

**What this solves.** Phases 1–4 filled the generator slot with a hosted API. The
fine-tuned model from Phase 4 lives on the Hub as weights, which is not a thing
anything can call. This turns those weights into a URL.

**Why OpenAI-compatible rather than a bespoke endpoint.** `common.generator`
already speaks that protocol via `OpenAICompatGenerator`, and every call in the
system goes through it (CLAUDE.md §8). Matching the protocol means the Phase 5
model swap is a *config change* — a new provider block pointing at this URL — and
not a single line of new Python anywhere upstream. A bespoke JSON shape would
have been marginally simpler to write here and would have forced a new generator
class, a new code path in the graph, and a second thing to keep in sync. This is
what the "swappable slot" in CLAUDE.md §3 was for; this file is the invoice.

**Why transformers and not vLLM.** CLAUDE.md §4 bans vLLM for ZeroGPU reasons.
That specific reason does not apply on Modal — but the ban is kept anyway, for a
different one: the serving path should stay identical whichever host we land on,
and `transformers` + `TextIteratorStreamer` is the only stack that runs on both.
Throughput is irrelevant at our traffic (one demo user at a time); portability is
not.

**Why the model class is discovered rather than declared.** P-20 and P-21 both
cost multiple launches because the *harness* failed in a way that read as the
*subject* failing. Qwen3.5-4B is multimodal, and the correct auto-class for a
merged multimodal checkpoint is not obvious across transformers versions. So the
loader tries the classes in order, prints which one won, and prints the resolved
library versions before it tries anything. A skew or a wrong class is then
visible in the first ten log lines instead of being reported as "the model is
broken".

**Cost.** Serverless: `min_containers=0`, so an idle demo costs nothing. A
container holds the GPU for `SCALEDOWN_WINDOW` seconds after the last request so
that a conversation does not pay a cold start per turn. Cold start is the real
cost driver — see `estimate` below and run it before deploying.

Usage:
    modal run deployment/modal_inference.py::estimate
    modal serve deployment/modal_inference.py     # ephemeral, for testing
    modal deploy deployment/modal_inference.py    # persistent URL
"""

# NOTE: no `from __future__ import annotations` here, and that omission is
# load-bearing. It would turn every annotation into a string, and FastAPI
# resolves annotation strings against the *module* globals — but `Request` is
# imported inside `api()`, so it would be invisible there and FastAPI would
# treat `request: Request` as a missing query parameter, returning
# `{"loc":["query","request"],"msg":"Field required"}` for every call. Kept out
# alongside the module-level fastapi import below — either alone fixes the bug,
# and both together mean no future edit can reintroduce it by accident.
# Python 3.11 evaluates `str | None` and `dict[str, Any]` natively, so nothing
# in this file needs postponed evaluation.

import json
import os
import time
import uuid
from typing import Any

import modal

# Bumped on every change to the request-handling code, and reported by /health.
# The reason it exists: the first fix for the `{"loc":["query","request"]}` bug
# was correct and appeared not to work, because there was no way to tell a wrong
# fix from a stale container still serving the old code. That is P-20's lesson
# again — a diagnostic must distinguish "the subject failed" from "I never
# reached the subject" — and it costs one string to never have that doubt again.
BUILD = "6"

# Imported at MODULE level, not inside `api()`, and that placement is the actual
# fix. FastAPI resolves a route's annotations with `typing.get_type_hints`, which
# looks them up in the function's `__globals__` — the module namespace. A
# `Request` imported inside `api()` is a local, invisible there, so FastAPI
# cannot tell `request: Request` is the request object and demotes it to a
# missing query parameter. At module level it resolves under every annotation
# regime, with or without postponed evaluation.
#
# Guarded because Modal imports this file twice: once on your machine to build
# the app graph (where fastapi may not be installed) and once inside the
# container (where it always is). The guard lets the client-side import succeed;
# the container never takes the fallback branch.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # pragma: no cover - client-side import only
    FastAPI = Request = JSONResponse = StreamingResponse = None  # type: ignore[assignment,misc]

# --- Configuration -----------------------------------------------------------
# Not read from config.yaml: this module is executed by Modal's CLI inside a
# container that has no repo checkout, so it cannot import `common.config`.
# These are deployment constants, and the ones a caller needs are overridable by
# environment variable at deploy time.

APP_NAME = "claimwise-inference"
MODEL_REPO = os.environ.get("CLAIMWISE_MODEL_REPO", "AbhiCommits/claimwise-qwen35-4b")

# L4 because that is what Phase 4 trained on and what the 12.49 GB peak was
# measured against (D-32). Inference needs less than training — no optimiser
# state, no gradients — so a 24 GB L4 is comfortable and is the cheapest card
# that fits the model in bf16 without quantisation.
GPU_KIND = "L4"

# How long a container keeps the GPU after its last request. The trade:
# too short and every turn of a conversation pays a cold start; too long and an
# abandoned browser tab bills for idle GPU. 120s covers a user thinking between
# questions and bounds the waste of a walk-away at two minutes.
SCALEDOWN_WINDOW = 120

# One container, several concurrent requests. Generation is GPU-bound, so this
# does not multiply throughput — it stops a second user's request from cold-
# starting a whole new container while the first is mid-answer.
MAX_CONCURRENT = 4

# Hard ceiling on generated tokens regardless of what a caller asks for. A demo
# endpoint on metered GPU must not let one request run away; CLAUDE.md §3 makes
# the Modal balance a hard cap, so the cap belongs in the server, not in the
# client's good manners.
MAX_NEW_TOKENS_CEILING = 2048

# Weights cached across cold starts. The download is ~8 GB and is most of a cold
# start's billed seconds; paying it once per deployment rather than once per
# container is the single biggest cost lever in this file.
HF_CACHE = modal.Volume.from_name("claimwise-hf-cache", create_if_missing=True)
CACHE_PATH = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Unpinned, and that is deliberate — P-20. Three hand-picked pins across
    # torch / transformers / accelerate must stay mutually consistent across
    # releases, and when they drifted the failure surfaced as an unrelated
    # `torch.utils._pytree` AttributeError that read as a model problem. Let pip
    # resolve one consistent stack; `/health` reports what it resolved.
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "fastapi[standard]",
        "hf_transfer",
        "huggingface_hub",
    )
    .env(
        {
            # Fast Hub downloads; ~8 GB is most of a cold start.
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
        }
    )
)

app = modal.App(APP_NAME)

# The shared secret the HF Space presents. Create it once with:
#   modal secret create claimwise-api CLAIMWISE_API_KEY=<a long random string>
# Never committed, never logged, never sent to the browser (CLAUDE.md §5.5 —
# the token lives server-side in the Space, not in the frontend bundle).
api_secret = modal.Secret.from_name("claimwise-api")


def _openai_error(status: int, message: str, kind: str) -> Any:
    """Build an error response the OpenAI SDK will surface intelligibly.

    The client is the real `openai` package (`common.generator`), which reads
    `error.message` off this shape. Returning a bare string here would surface
    upstream as an unhelpful generic status error, which is exactly the kind of
    "the harness obscured the failure" problem P-20 and P-21 were about.

    Args:
        status: HTTP status code to return.
        message: Human-readable explanation.
        kind: OpenAI error `type` field, e.g. "invalid_request_error".

    Returns:
        A FastAPI JSON response carrying the error.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": kind, "code": status}},
    )


@app.cls(
    image=image,
    gpu=GPU_KIND,
    volumes={CACHE_PATH: HF_CACHE},
    secrets=[api_secret],
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=0,
    timeout=600,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class Server:
    """The model, loaded once per container, behind a FastAPI app."""

    @modal.enter()
    def load(self) -> None:
        """Load the tokenizer and model onto the GPU.

        Runs once per container, not once per request — this is the whole reason
        `@modal.enter` exists, and getting it wrong would reload 8 GB of weights
        on every question.

        Raises:
            RuntimeError: If no known auto-class can load the checkpoint. Raised
                loudly with the attempts listed, because a silent fallback to a
                wrong class is how a multimodal checkpoint ends up served as a
                text model with its vision tower quietly missing.
        """
        import torch
        import transformers
        from transformers import AutoProcessor, AutoTokenizer

        # Printed BEFORE anything is attempted — the Check 0 lesson from P-20.
        # When a load fails, the first question is always "what was actually
        # installed", and the first probe left no way to answer it.
        print(
            f"versions: torch={torch.__version__} "
            f"transformers={transformers.__version__} "
            f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}"
        )
        print(f"loading  : {MODEL_REPO}")
        started = time.perf_counter()

        # A processor for a multimodal checkpoint, a tokenizer for a text one.
        # `getattr(processor, "tokenizer", processor)` is the P-20 fix: the
        # wrapper's own __call__ takes (images=, text=) and reads a positional
        # list of strings as image sources.
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_REPO)
            print("processor: AutoProcessor (multimodal)")
        except Exception as error:  # noqa: BLE001 - fall back, but say why
            print(f"processor: AutoProcessor failed ({error}); using AutoTokenizer")
            self.processor = AutoTokenizer.from_pretrained(MODEL_REPO)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Tried in order, most specific first. Discovery rather than declaration,
        # because the right class for a merged Qwen3.5 checkpoint has moved
        # between transformers releases and a wrong guess here is a failed
        # deployment that looks like a broken model.
        attempts: list[str] = []
        self.model = None
        for class_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            auto_class = getattr(transformers, class_name, None)
            if auto_class is None:
                attempts.append(f"{class_name}: not in this transformers version")
                continue
            try:
                self.model = auto_class.from_pretrained(
                    MODEL_REPO, dtype=torch.bfloat16, device_map="cuda"
                )
                print(f"model    : loaded via {class_name}")
                break
            except Exception as error:  # noqa: BLE001 - try the next class
                attempts.append(f"{class_name}: {error}")

        if self.model is None:
            raise RuntimeError(
                "No auto-class could load "
                f"{MODEL_REPO}. Attempts:\n  " + "\n  ".join(attempts)
            )

        self.model.eval()
        print(f"ready    : {time.perf_counter() - started:.1f}s cold start")

    def _render(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model's chat template to OpenAI-shaped messages.

        Args:
            messages: OpenAI `messages`, each with `role` and `content`.

        Returns:
            The prompt string with the assistant turn opened.
        """
        # `enable_thinking=False` because Qwen3.5's template turns reasoning on
        # by DEFAULT, and the first live call proved it: the model answered with
        # "Here's a thinking process that leads to the suggested answer:" and
        # then reasoned out loud instead of answering. That is a train/serve
        # mismatch — the RAFT set (Phase 4) is `(question + context) -> grounded
        # cited answer` with no thinking blocks, so serving with thinking on
        # asks the model for a format it was never tuned to produce, and leaks
        # the reasoning into the user's answer either way.
        #
        # Passed defensively: the kwarg is Qwen-specific and older templates
        # raise TypeError on an unknown argument. Falling back keeps a template
        # change from taking the endpoint down, and the fallback is logged so a
        # silently-thinking model is never mistaken for a badly-tuned one.
        try:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            print("warn     : template rejected enable_thinking=False; thinking may be ON")
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _generation_kwargs(self, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Build `model.generate` arguments from OpenAI request fields.

        Temperature 0 becomes greedy decoding rather than a tiny temperature.
        The evals depend on reproducibility (CLAUDE.md §8), and `temperature=0.0`
        passed to a sampler is undefined behaviour that some stacks silently
        clamp to 1e-7 and others divide by.

        Args:
            temperature: OpenAI temperature. 0.0 means greedy.
            max_tokens: Requested completion cap, before the server ceiling.

        Returns:
            Keyword arguments for `generate`.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": min(int(max_tokens), MAX_NEW_TOKENS_CEILING),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    @modal.asgi_app()
    def api(self) -> Any:
        """Build the OpenAI-compatible FastAPI app.

        Returns:
            The ASGI application Modal will serve.
        """
        web = FastAPI(title="ClaimWise inference", version=BUILD)

        def authorised(request: Request) -> bool:
            """Check the bearer token against the Modal secret.

            Args:
                request: The incoming request.

            Returns:
                True when the token matches.
            """
            expected = os.environ.get("CLAIMWISE_API_KEY", "")
            header = request.headers.get("authorization", "")
            token = header[7:] if header.lower().startswith("bearer ") else ""
            # Constant-time compare: a plain `==` leaks the shared secret one
            # byte at a time to anyone who can measure response timing.
            import hmac

            return bool(expected) and hmac.compare_digest(token, expected)

        @web.get("/")
        def root() -> dict[str, Any]:
            """Name the available routes so a browser hit is not a bare 404.

            Returns:
                The service name and its routes.
            """
            return {
                "service": "claimwise-inference",
                "build": BUILD,
                "routes": ["/health", "/v1/models", "/v1/chat/completions"],
            }

        @web.get("/health")
        def health() -> dict[str, Any]:
            """Report readiness and the resolved stack. No auth — no secrets here.

            Returns:
                Model id, device, and library versions.
            """
            import torch
            import transformers

            return {
                "status": "ok",
                "build": BUILD,
                "model": MODEL_REPO,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "model_class": type(self.model).__name__,
            }

        @web.get("/v1/models")
        def models(request: Request) -> Any:
            """List the served model, in the shape the OpenAI SDK expects.

            Args:
                request: The incoming request, for auth.

            Returns:
                An OpenAI model list.
            """
            if not authorised(request):
                return _openai_error(401, "Invalid API key.", "authentication_error")
            return {
                "object": "list",
                "data": [{"id": MODEL_REPO, "object": "model", "owned_by": "claimwise"}],
            }

        @web.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            """Generate a completion, streaming or not.

            Args:
                request: An OpenAI chat-completions request.

            Returns:
                An OpenAI completion object, or an SSE stream of chunks.
            """
            if not authorised(request):
                return _openai_error(401, "Invalid API key.", "authentication_error")

            body = await request.json()
            messages = body.get("messages") or []
            if not messages:
                return _openai_error(400, "`messages` is required.", "invalid_request_error")

            kwargs = self._generation_kwargs(
                temperature=body.get("temperature", 0.0),
                max_tokens=body.get("max_tokens", 1024),
            )
            prompt = self._render(messages)
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            if body.get("stream"):
                return StreamingResponse(
                    self._stream(prompt, kwargs, completion_id, created),
                    media_type="text/event-stream",
                )

            text, prompt_tokens, completion_tokens = self._complete(prompt, kwargs)
            # "length" when the cap was hit, "stop" only when the model chose to
            # end. Reporting a truncated answer as "stop" would put a silently
            # cut-off completion into METRICS as a complete one (CLAUDE.md §6),
            # and truncation removes the END of an answer — which for this model
            # is where the page citation sits.
            finish = "length" if completion_tokens >= kwargs["max_new_tokens"] else "stop"
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_REPO,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                ],
                # `common.generator` reads these straight into GenerationResult,
                # which is where "tokens per query" in METRICS.md comes from. An
                # endpoint that omitted usage would silently zero that metric.
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        return web

    def _complete(self, prompt: str, kwargs: dict[str, Any]) -> tuple[str, int, int]:
        """Generate a full completion and count tokens.

        Args:
            prompt: The rendered prompt.
            kwargs: Arguments for `generate`.

        Returns:
            `(text, prompt_tokens, completion_tokens)`.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        # Slice off the prompt: `generate` returns prompt + completion, and
        # decoding the whole thing would echo the entire retrieved context back
        # to the user as if the model had written it.
        new_tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), prompt_tokens, int(new_tokens.shape[-1])

    def _stream(
        self,
        prompt: str,
        kwargs: dict[str, Any],
        completion_id: str,
        created: int,
    ) -> Any:
        """Yield OpenAI-format SSE chunks as tokens are produced.

        `TextIteratorStreamer` requires `generate` to run on another thread —
        it is a blocking call, and the streamer is drained from this one.

        Args:
            prompt: The rendered prompt.
            kwargs: Arguments for `generate`.
            completion_id: Stable id repeated in every chunk.
            created: Unix timestamp repeated in every chunk.

        Yields:
            `data: {...}` SSE lines, terminated by `data: [DONE]`.
        """
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        import threading

        def run() -> None:
            """Generate on a worker thread so the streamer can be drained."""
            with torch.inference_mode():
                self.model.generate(**inputs, **kwargs, streamer=streamer)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def chunk(delta: dict[str, Any], finish: str | None) -> str:
            """Format one SSE line.

            Args:
                delta: The incremental message payload.
                finish: Finish reason, or None mid-stream.

            Returns:
                A complete `data:` line.
            """
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_REPO,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        yield chunk({"role": "assistant", "content": ""}, None)
        for token in streamer:
            if token:
                yield chunk({"content": token}, None)
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"


@app.local_entrypoint()
def estimate() -> None:
    """Print the cost model before anything is deployed.

    CLAUDE.md §3 requires an estimated cost printed before a real Modal run, and
    §6 requires the remaining balance tracked. Serving is not a single job with
    one price, so what this prints is the *rate card and the drivers* — the two
    numbers that actually decide the bill are cold starts and idle window, and
    both are choices made in this file rather than facts about the workload.
    """
    # Read off modal.com/pricing at deploy time and corrected here. METRICS.md
    # already flags that every Phase 4 cost estimate depends on an UNVERIFIED
    # $0.80/L4-hour rate; do not let a second table inherit the same guess.
    rate_per_hour = float(os.environ.get("CLAIMWISE_L4_RATE", "0.80"))
    per_second = rate_per_hour / 3600.0

    cold_start_s = 90.0   # weight load from the Volume; measure and replace
    per_query_s = 6.0     # generation only, warm container; measure and replace

    print("=== ClaimWise inference — cost model ===")
    print(f"  GPU                  : {GPU_KIND} @ ${rate_per_hour:.2f}/hour (UNVERIFIED)")
    print(f"  model                : {MODEL_REPO}")
    print(f"  idle cost            : $0.00  (min_containers=0)")
    print(f"  cold start           : ~{cold_start_s:.0f}s = ${cold_start_s * per_second:.4f}")
    print(f"  warm query           : ~{per_query_s:.0f}s = ${per_query_s * per_second:.4f}")
    print(f"  scaledown window     : {SCALEDOWN_WINDOW}s = "
          f"${SCALEDOWN_WINDOW * per_second:.4f} billed after the last request")
    print()
    session = cold_start_s + 10 * per_query_s + SCALEDOWN_WINDOW
    print(f"  10-question session  : ~{session:.0f}s = ${session * per_second:.4f}")
    print(f"  100 such sessions    : ${100 * session * per_second:.2f}")
    print()
    print("  The cold start and warm-query figures above are ASSUMPTIONS.")
    print("  Read the real ones off `modal app logs` after the first deploy and")
    print("  replace them here, then record the rate in METRICS.md.")
