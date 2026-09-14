"""An OpenAI-compatible endpoint in front of one streaming engine.

Greedy by default: `temperature: 0` is `argmax`. Sampling, the penalties,
`logit_bias`, `stop` and `n` are honoured; a value outside its range is
refused rather than clamped.
"""

import json
import logging
import queue
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mlx_lean_moe.runtime.sampling import Sampling

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_CONTEXT = 32768


_PYTORCH_NOTICE = "PyTorch was not found"


def _hush_missing_pytorch() -> None:
    """Silence transformers' import-time warning about PyTorch being absent;
    only its tokenizer is used here."""
    logging.getLogger("transformers").addFilter(
        lambda record: _PYTORCH_NOTICE not in record.getMessage()
    )


class Engine:
    """One loaded checkpoint, driven from one worker thread: MLX's default
    stream is thread-local, and that worker also serialises requests."""

    def __init__(
        self,
        model_dir: str | Path,
        model_id: str,
        *,
        max_context: int,
        expert_cache_size_per_layer: int | None = None,
        think: bool = False,
    ) -> None:
        self.model_id = model_id
        self.think = think
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-lean-moe"
        )

        def build() -> None:
            import json as _json

            _hush_missing_pytorch()
            from transformers import AutoTokenizer

            from mlx_lean_moe.config import model_config_from_hf
            from mlx_lean_moe.runtime.generate import ChatSession
            from mlx_lean_moe.weights.expert_loader import decode_on_this_thread

            decode_on_this_thread()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            self.stop_ids = _stop_token_ids(Path(model_dir), self.tokenizer)
            config = model_config_from_hf(
                _json.loads((Path(model_dir) / "config.json").read_text())
            )
            self.session = ChatSession(
                model_dir, config, max_context, expert_cache_size_per_layer
            )

        self._worker.submit(build).result()
        self.max_context = max_context

    def context_error(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> str | None:
        """Why this request will not fit, checked before a byte is answered:
        a stream has sent its headers by the time the model would raise."""
        prompt = len(self.prompt_ids(messages))
        if prompt + max_tokens <= self.max_context:
            return None
        return (
            f"this request needs {prompt} prompt tokens plus {max_tokens} of answer, "
            f"over the server's context of {self.max_context}. Shorten it, lower max_tokens, "
            f"or restart with --max-context above {prompt + max_tokens}"
        )

    def close(self) -> None:
        self._worker.submit(self.session.close).result()
        self._worker.shutdown(wait=True)

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Render a conversation to prompt token ids. Templates disagree on
        what they do when nobody asks for thinking, so it is always passed."""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                enable_thinking=self.think,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: list[str],
        sampling: Sampling | None = None,
    ) -> Iterator[tuple[str, str | None]]:
        """Run a generation on the worker and hand its pieces back through a
        bounded queue, so a slow client stops the model."""
        pieces: queue.Queue = queue.Queue(maxsize=8)
        done = object()

        def run() -> None:
            try:
                for piece in self._generate(messages, max_tokens, stop, sampling):
                    pieces.put(piece)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                pieces.put(exc)
            finally:
                pieces.put(done)

        self._worker.submit(run)
        while True:
            piece = pieces.get()
            if piece is done:
                return
            if isinstance(piece, BaseException):
                raise piece
            yield piece

    def _generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: list[str],
        sampling: Sampling | None = None,
    ) -> Iterator[tuple[str, str | None]]:
        """Yield ``(text, finish_reason)``, holding back the last
        ``len(longest stop) - 1`` characters: "a#" cannot be taken back."""
        prompt = self.prompt_ids(messages)
        hold = max((len(s) for s in stop if s), default=1) - 1

        pending: list[int] = []
        produced = ""
        sent = 0
        finish = "length"

        for token in self.session.send(
            prompt, max_new_tokens=max_tokens, sampling=sampling
        ):
            if token in self.stop_ids:
                finish = "stop"
                break

            pending.append(token)
            text = self.tokenizer.decode(pending)
            if "�" in text:
                continue
            pending.clear()

            produced += text
            hit = _first_stop(produced, stop)
            if hit is not None:
                if hit > sent:
                    yield produced[sent:hit], None
                yield "", "stop"
                return

            safe = len(produced) - hold
            if safe > sent:
                yield produced[sent:safe], None
                sent = safe

        if len(produced) > sent:
            yield produced[sent:], None
        yield "", finish


def _stop_token_ids(model_dir: Path, tokenizer) -> set[int]:
    """Every token id that ends a turn: some checkpoints publish several
    beyond ``tokenizer.eos_token_id``."""
    generation_config = model_dir / "generation_config.json"
    if generation_config.exists():
        end = json.loads(generation_config.read_text()).get("eos_token_id")
        if end is not None:
            return {end} if isinstance(end, int) else set(end)
    return {tokenizer.eos_token_id}


def _first_stop(text: str, stop: list[str]) -> int | None:
    """Where the earliest stop sequence begins, if one has appeared."""
    found = [text.find(s) for s in stop if s]
    hits = [at for at in found if at >= 0]
    return min(hits) if hits else None


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish: str | None,
    index: int = 0,
) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _logit_bias(raw: Any) -> dict[int, float]:
    """JSON object keys are strings, so the token ids arrive as text."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("logit_bias must be an object of token id to bias")
    bias = {}
    for token, value in raw.items():
        try:
            bias[int(token)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"logit_bias[{token!r}] must be a number keyed by a token id"
            ) from exc
    return bias


def _choice_count(raw: Any) -> int:
    if raw is None:
        return 1
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"n must be an integer, got {raw!r}")
    if raw < 1:
        raise ValueError(f"n must be at least 1, got {raw}")
    return raw


class _Handler(BaseHTTPRequestHandler):
    engine: Engine  # set on the bound subclass `serve` makes

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib's name
        pass

    def _json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json(
            status, {"error": {"message": message, "type": "invalid_request_error"}}
        )

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_id,
                            "object": "model",
                            "owned_by": "mlx-lean-moe",
                        }
                    ],
                },
            )
            return
        self._error(404, f"no route for {self.path}")

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"no route for {self.path}")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f"could not read the request body: {exc}")
            return

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "messages must be a non-empty list")
            return

        max_tokens = (
            request.get("max_tokens")
            or request.get("max_completion_tokens")
            or DEFAULT_MAX_TOKENS
        )
        stop = request.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]

        try:
            choices = _choice_count(request.get("n"))
            sampling = Sampling(
                temperature=float(request.get("temperature") or 0.0),
                top_p=float(
                    request.get("top_p") if request.get("top_p") is not None else 1.0
                ),
                seed=request.get("seed"),
                frequency_penalty=float(request.get("frequency_penalty") or 0.0),
                presence_penalty=float(request.get("presence_penalty") or 0.0),
                logit_bias=_logit_bias(request.get("logit_bias")),
            )
        except (TypeError, ValueError) as exc:
            self._error(400, str(exc))
            return

        too_long = self.engine.context_error(messages, int(max_tokens))
        if too_long is not None:
            self._error(400, too_long)
            return

        try:
            if request.get("stream"):
                self._stream(messages, int(max_tokens), stop, sampling, choices)
            else:
                self._whole(messages, int(max_tokens), stop, sampling, choices)
        except BrokenPipeError:
            pass

    def _complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: list[str],
        sampling: Sampling,
        index: int,
    ) -> tuple[str, str]:
        text, finish = "", "stop"
        for piece, reason in self.engine.generate(
            messages, max_tokens, stop, sampling.for_choice(index)
        ):
            text += piece
            if reason is not None:
                finish = reason
        return text, finish

    def _whole(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: list[str],
        sampling: Sampling,
        choices: int = 1,
    ) -> None:
        answers = [
            self._complete(messages, max_tokens, stop, sampling, index)
            for index in range(choices)
        ]
        self._json(
            200,
            {
                "id": _completion_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.engine.model_id,
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                    for index, (text, finish) in enumerate(answers)
                ],
            },
        )

    def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: list[str],
        sampling: Sampling,
        choices: int = 1,
    ) -> None:
        completion_id = _completion_id()
        model = self.engine.model_id
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def write(payload: str) -> None:
            self.wfile.write(payload.encode())
            self.wfile.flush()

        for index in range(choices):
            write(
                _chunk(
                    completion_id,
                    model,
                    {"role": "assistant", "content": ""},
                    None,
                    index,
                )
            )
            pieces = self.engine.generate(
                messages, max_tokens, stop, sampling.for_choice(index)
            )
            for piece, reason in pieces:
                if piece:
                    write(_chunk(completion_id, model, {"content": piece}, None, index))
                if reason is not None:
                    write(_chunk(completion_id, model, {}, reason, index))
        write("data: [DONE]\n\n")
        self.close_connection = True


class _Server(ThreadingHTTPServer):
    """A server that does not mistake a client leaving for a fault: the
    stdlib prints a traceback for a disconnect the same as for a bug."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(
            sys.exception(), (ConnectionError, BrokenPipeError, TimeoutError)
        ):
            return
        super().handle_error(request, client_address)


def serve(
    engine: Engine, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Start serving `engine`. Returns the server, already listening."""
    handler = type("_BoundHandler", (_Handler,), {"engine": engine})
    return _Server((host, port), handler)


def _resolve_model(requested: str | None) -> str:
    """The repo id to load: what was asked for, or the cache's own answer when
    it holds exactly one checkpoint."""
    from mlx_lean_moe.weights import download

    if requested is not None:
        if download.is_complete(requested):
            return requested
        raise SystemExit(
            f"{requested} is not downloaded. Fetch it with:\n  python -m mlx_lean_moe.weights.download {requested}"
        )

    cached = download.cached_checkpoints()
    if len(cached) == 1:
        return cached[0]
    if not cached:
        raise SystemExit(
            "no checkpoint is downloaded. Fetch one with:\n  python -m mlx_lean_moe.weights.download <repo id>"
        )
    listed = "\n".join(f"  {repo_id}" for repo_id in cached)
    raise SystemExit(
        f"several checkpoints are downloaded; pick one with --model:\n{listed}"
    )


def _main() -> None:
    import argparse
    import sys

    from mlx_lean_moe.weights import download

    parser = argparse.ArgumentParser(
        description="Serve one checkpoint over an OpenAI-compatible endpoint. "
        "Point aider, or anything else that speaks that API, at the printed base URL.",
    )
    parser.add_argument(
        "--model", default=None, help="hub repo id (default: the only one downloaded)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=f"attention context length, fixed at startup (default: {DEFAULT_MAX_CONTEXT})",
    )
    parser.add_argument(
        "--expert-cache",
        type=int,
        default=None,
        help="experts kept resident per layer (default: the top-k). The memory dial.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="let the model emit a reasoning block before answering",
    )
    arguments = parser.parse_args()

    model = _resolve_model(arguments.model)
    engine = Engine(
        download.snapshot_dir(model),
        model,
        max_context=arguments.max_context,
        expert_cache_size_per_layer=arguments.expert_cache,
        think=arguments.think,
    )
    server = serve(engine, arguments.host, arguments.port)
    host, port = server.server_address[:2]
    print(f"serving {model} at http://{host}:{port}/v1", file=sys.stderr)
    print(
        f"  aider --openai-api-base http://{host}:{port}/v1 --model openai/{model}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    _main()
