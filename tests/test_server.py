"""The OpenAI-compatible endpoint, against a fake engine: the protocol is
what is under test, not the model.

A real server on a real port, since chunked streaming and header flushing
do not exist in a hand-called handler.
"""

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlx_lean_moe.server import _first_stop, _resolve_model, serve


class FakeEngine:
    """Yields what it was told to, in (text, finish_reason) pairs."""

    def __init__(self, pieces=("Hello", ", ", "world"), finish="stop") -> None:
        self.model_id = "fake/model"
        self.max_context = 8192
        self.pieces = pieces
        self.finish = finish
        self.seen: list[dict] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def context_error(self, messages, max_tokens):
        return None

    def generate(self, messages, max_tokens, stop, sampling=None):
        self.seen.append({"messages": messages, "max_tokens": max_tokens, "stop": stop, "sampling": sampling})
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            for piece in self.pieces:
                time.sleep(0.01)  # long enough for a second request to overlap if it could
                yield piece, None
            yield "", self.finish
        finally:
            with self._lock:
                self.concurrent -= 1


@pytest.fixture
def server():
    engine = FakeEngine()
    httpd = serve(engine, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, engine, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(port, body, path="/v1/chat/completions"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return connection, connection.getresponse()


def test_models_lists_the_one_checkpoint(server):
    _, engine, port = server
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    body = json.loads(connection.getresponse().read())
    assert body["data"][0]["id"] == engine.model_id
    connection.close()


def test_a_whole_completion_has_the_shape_clients_expect(server):
    _, engine, port = server
    messages = [{"role": "user", "content": "hi"}]
    connection, response = _post(port, {"messages": messages, "max_tokens": 7})
    body = json.loads(response.read())
    connection.close()

    assert response.status == 200
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello, world"}
    assert choice["finish_reason"] == "stop"
    assert body["model"] == engine.model_id
    # The request's own limits reach the engine rather than being dropped.
    assert engine.seen[0]["max_tokens"] == 7
    assert engine.seen[0]["messages"] == messages


def test_streaming_sends_deltas_and_terminates(server):
    """Clients read this incrementally, so the framing is the contract: one
    `data:` line per chunk, a blank line between, and `[DONE]` at the end."""
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    raw = response.read().decode()
    connection.close()

    assert response.getheader("Content-Type") == "text/event-stream"
    assert raw.endswith("data: [DONE]\n\n")

    events = [line[len("data: ") :] for line in raw.split("\n\n") if line.startswith("data: ")]
    parsed = [json.loads(e) for e in events if e != "[DONE]"]
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(p["choices"][0]["delta"].get("content", "") for p in parsed) == "Hello, world"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
    assert {p["object"] for p in parsed} == {"chat.completion.chunk"}


def test_max_tokens_defaults_when_the_client_names_none(server):
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}]})
    response.read()
    connection.close()
    assert engine.seen[0]["max_tokens"] > 0


def test_a_string_stop_is_taken_as_a_list(server):
    """The API allows either, and clients use both."""
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "stop": "###"})
    response.read()
    connection.close()
    assert engine.seen[0]["stop"] == ["###"]


def test_concurrent_requests_all_get_answered(server):
    """Three clients at once must each get a whole answer, serialised by the
    engine's single worker."""
    _, engine, port = server
    answers: list[str] = []
    errors: list[Exception] = []

    def ask():
        try:
            connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}]})
            answers.append(json.loads(response.read())["choices"][0]["message"]["content"])
            connection.close()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors
    assert answers == ["Hello, world"] * 3


def test_a_request_without_messages_is_refused(server):
    _, _, port = server
    for body in ({}, {"messages": []}, {"messages": "hello"}):
        connection, response = _post(port, body)
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 400
        assert "messages" in payload["error"]["message"]


def test_an_unknown_route_is_a_404(server):
    _, _, port = server
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/embeddings")
    assert connection.getresponse().status == 404
    connection.close()


def test_malformed_json_is_refused_rather_than_crashing(server):
    _, _, port = server
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/v1/chat/completions", "{not json", {"Content-Type": "application/json"})
    response = connection.getresponse()
    assert response.status == 400
    response.read()
    connection.close()


@pytest.mark.parametrize(
    ("text", "stop", "expected"),
    [
        ("hello world", ["world"], 6),
        ("hello world", ["zzz"], None),
        ("a b c", ["c", "b"], 2),  # the earliest, not the first named
        ("hello", [], None),
        ("hello", [""], None),  # an empty stop matches everywhere; it must not
    ],
)
def test_first_stop_finds_the_earliest_sequence(text, stop, expected):
    assert _first_stop(text, stop) == expected


class _Tokenizer:
    """Decodes ids as characters, so a test can spell out what the model
    'said'. Id 0 is end-of-sequence."""

    eos_token_id = 0

    def __init__(self, vocabulary: dict[int, str]) -> None:
        self.vocabulary = vocabulary

    def decode(self, ids):
        return "".join(self.vocabulary.get(i, "�") for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class _Session:
    def __init__(self, tokens) -> None:
        self.tokens = tokens
        self.sent: list[list[int]] = []

    def send(self, token_ids, max_new_tokens, sampling=None):
        self.sent.append(list(token_ids))
        self.sampling = sampling
        yield from self.tokens[:max_new_tokens]


@pytest.fixture
def make_engine():
    """A real Engine with its model replaced, and a real worker thread: the
    decoding, stop handling and hand-off are Engine's own."""
    built = []

    def make(tokens, vocabulary):
        from mlx_lean_moe.server import Engine

        engine = Engine.__new__(Engine)
        engine.model_id = "test/model"
        engine.think = False
        engine.tokenizer = _Tokenizer(vocabulary)
        engine.stop_ids = {0}
        engine.session = _Session(tokens)
        engine._worker = ThreadPoolExecutor(max_workers=1)
        built.append(engine)
        return engine

    yield make
    for engine in built:
        engine._worker.shutdown(wait=True)


def _collect(engine, **kwargs):
    pieces, finish = [], None
    for text, reason in engine.generate([{"role": "user", "content": "hi"}], **kwargs):
        pieces.append(text)
        if reason is not None:
            finish = reason
    return "".join(pieces), finish


def test_engine_stops_at_end_of_sequence_without_emitting_it(make_engine):
    engine = make_engine([5, 6, 0, 7], {5: "h", 6: "i", 7: "!"})
    assert _collect(engine, max_tokens=10, stop=[]) == ("hi", "stop")


def test_engine_reports_length_when_it_runs_out_of_budget(make_engine):
    engine = make_engine([5, 6, 7], {5: "h", 6: "i", 7: "!"})
    assert _collect(engine, max_tokens=2, stop=[]) == ("hi", "length")


def test_engine_holds_back_a_half_decoded_character(make_engine):
    """One token can be half a multi-byte character. Emitting it would show
    the client a replacement glyph and claim the model produced it."""
    engine = make_engine([5, 6, 7], {5: "h", 7: "!"})  # id 6 decodes to U+FFFD alone
    text, _ = _collect(engine, max_tokens=10, stop=[])
    assert "�" not in text


def test_engine_cuts_the_answer_at_a_stop_sequence(make_engine):
    engine = make_engine([5, 6, 7, 8], {5: "a", 6: "#", 7: "#", 8: "b"})
    assert _collect(engine, max_tokens=10, stop=["##"]) == ("a", "stop")


def test_engine_gives_the_session_the_whole_conversation(make_engine):
    """The session reuses whatever prefix it already holds, so it wants the
    full context every turn rather than only what is new."""
    engine = make_engine([0], {})
    _collect(engine, max_tokens=4, stop=[])
    assert engine.session.sent == [[1, 2, 3]]


def test_engine_does_not_stream_the_start_of_a_stop_sequence(make_engine):
    """Tokens "a", "#", "#" against a "##" stop: sending the first "#" as it
    arrives puts half the forbidden sequence in front of the client."""
    engine = make_engine([5, 6, 7, 8], {5: "a", 6: "#", 7: "#", 8: "b"})
    pieces = [text for text, _ in engine.generate([{"role": "user", "content": "hi"}], 10, ["##"])]
    assert "".join(pieces) == "a"
    assert all("#" not in piece for piece in pieces)


def test_engine_releases_the_held_tail_when_no_stop_arrives(make_engine):
    """Holding back the tail must not lose it: text kept in case it began a
    stop sequence is still the model's answer when the sequence never comes."""
    engine = make_engine([5, 6, 7, 0], {5: "a", 6: "#", 7: "b"})
    assert _collect(engine, max_tokens=10, stop=["##"]) == ("a#b", "stop")


def test_engine_holds_nothing_back_without_stop_sequences(make_engine):
    """The common case: no stops asked for, so every character goes out as
    soon as it decodes rather than waiting behind a hold that cannot apply."""
    engine = make_engine([5, 6, 7], {5: "a", 6: "b", 7: "c"})
    pieces = [t for t, _ in engine.generate([{"role": "user", "content": "hi"}], 3, []) if t]
    assert pieces == ["a", "b", "c"]


def test_the_engine_runs_one_generation_at_a_time(make_engine):
    """The model lives on one worker thread, MLX's default stream being
    thread-local; that also keeps two answers from interleaving."""
    engine = make_engine([5, 6, 7, 0], {5: "a", 6: "b", 7: "c"})
    depth, peak, guard = 0, 0, threading.Lock()
    real_send = engine.session.send

    def counting_send(token_ids, max_new_tokens, sampling=None):
        nonlocal depth, peak
        with guard:
            depth += 1
            peak = max(peak, depth)
        try:
            for token in real_send(token_ids, max_new_tokens):
                time.sleep(0.01)
                yield token
        finally:
            with guard:
                depth -= 1

    engine.session.send = counting_send
    threads = [threading.Thread(target=lambda: _collect(engine, max_tokens=4, stop=[])) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert peak == 1


def test_the_engine_reraises_a_failure_from_its_worker(make_engine):
    """A generation that dies on the worker must surface on the thread that
    asked for it, not vanish into a queue nobody is watching."""
    engine = make_engine([5], {5: "a"})

    def exploding_send(token_ids, max_new_tokens, sampling=None):
        raise RuntimeError("the model fell over")
        yield  # pragma: no cover - makes this a generator

    engine.session.send = exploding_send
    with pytest.raises(RuntimeError, match="fell over"):
        _collect(engine, max_tokens=4, stop=[])


def test_every_published_end_of_turn_token_stops_generation(tmp_path):
    """Checkpoints can publish several; stopping on only the tokenizer's own
    lets generation run past a real stop."""
    from mlx_lean_moe.server import _stop_token_ids

    class _Tok:
        eos_token_id = 7

    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [11, 12]}))
    assert _stop_token_ids(tmp_path, _Tok()) == {11, 12}

    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 11}))
    assert _stop_token_ids(tmp_path, _Tok()) == {11}

    # No generation config, or one that names none: the tokenizer's own.
    (tmp_path / "generation_config.json").write_text(json.dumps({}))
    assert _stop_token_ids(tmp_path, _Tok()) == {7}
    (tmp_path / "generation_config.json").unlink()
    assert _stop_token_ids(tmp_path, _Tok()) == {7}


def test_a_second_end_of_turn_token_is_honoured(make_engine):
    engine = make_engine([5, 9, 6], {5: "a", 6: "b", 9: "<end>"})
    engine.stop_ids = {0, 9}
    assert _collect(engine, max_tokens=10, stop=[]) == ("a", "stop")


def test_a_client_leaving_is_not_reported_as_a_fault(server, capsys):
    """A client closing a keep-alive connection reaches the same handler a
    bug does, and a traceback per disconnect buries the real faults."""
    httpd, _, port = server

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/v1/models")
    connection.getresponse().read()
    connection.close()

    try:
        raise ConnectionResetError(54, "Connection reset by peer")
    except ConnectionResetError:
        httpd.handle_error(None, ("127.0.0.1", port))
    assert capsys.readouterr().err == ""

    try:
        raise ValueError("a real bug")
    except ValueError:
        httpd.handle_error(None, ("127.0.0.1", port))
    assert "a real bug" in capsys.readouterr().err


def test_sampling_parameters_reach_the_engine(server):
    """A parameter is either honoured or refused; dropping it silently
    leaves the client wrong about what it asked."""
    _, engine, port = server
    connection, response = _post(
        port,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 1234,
        },
    )
    response.read()
    connection.close()

    sampling = engine.seen[0]["sampling"]
    assert (sampling.temperature, sampling.top_p, sampling.seed) == (0.7, 0.9, 1234)
    assert not sampling.greedy


def test_a_request_naming_no_sampling_stays_greedy(server):
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}]})
    response.read()
    connection.close()
    assert engine.seen[0]["sampling"].greedy


def test_an_impossible_sampling_setting_is_refused(server):
    _, _, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "top_p": 1.5})
    payload = json.loads(response.read())
    connection.close()
    assert response.status == 400
    assert "top_p" in payload["error"]["message"]


def _cache(monkeypatch, complete, whole=()):
    monkeypatch.setattr("mlx_lean_moe.weights.download.cached_checkpoints", lambda *a, **k: list(complete))
    monkeypatch.setattr("mlx_lean_moe.weights.download.is_complete", lambda repo_id, *a, **k: repo_id in whole)


def test_one_downloaded_checkpoint_needs_no_flag(monkeypatch):
    _cache(monkeypatch, ["owner/only"])
    assert _resolve_model(None) == "owner/only"


def test_an_explicit_model_is_used_even_when_others_are_cached(monkeypatch):
    _cache(monkeypatch, ["owner/a", "owner/b"], whole=["owner/b"])
    assert _resolve_model("owner/b") == "owner/b"


def test_several_checkpoints_ask_rather_than_guess(monkeypatch):
    """Picking one would load 19 GB of whichever sorted first, which is not a
    guess worth making on the user's behalf."""
    _cache(monkeypatch, ["owner/a", "owner/b"])
    with pytest.raises(SystemExit) as raised:
        _resolve_model(None)
    assert "owner/a" in str(raised.value) and "owner/b" in str(raised.value)


def test_no_checkpoint_says_how_to_get_one(monkeypatch):
    _cache(monkeypatch, [])
    with pytest.raises(SystemExit, match="mlx_lean_moe.weights.download"):
        _resolve_model(None)


def test_an_undownloaded_model_is_refused_before_loading(monkeypatch):
    """Naming a repo id that is not there must not fall back to the one that
    is -- answers would come from a model the user did not ask for."""
    _cache(monkeypatch, ["owner/other"], whole=["owner/other"])
    with pytest.raises(SystemExit, match="owner/absent"):
        _resolve_model("owner/absent")


def test_penalties_and_bias_reach_the_engine(server):
    _, engine, port = server
    connection, response = _post(
        port,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5,
            "presence_penalty": -0.25,
            "logit_bias": {"7": -100, "9": 12.5},
        },
    )
    response.read()
    connection.close()

    sampling = engine.seen[0]["sampling"]
    assert sampling.frequency_penalty == 0.5
    assert sampling.presence_penalty == -0.25
    assert sampling.logit_bias == {7: -100.0, 9: 12.5}


def test_logit_bias_keys_arrive_as_strings_and_become_token_ids(server):
    """JSON has no integer keys, so a client's `{"7": -100}` must not end up
    indexing the vocabulary with the string."""
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "logit_bias": {"7": -100}})
    response.read()
    connection.close()
    assert list(engine.seen[0]["sampling"].logit_bias) == [7]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"frequency_penalty": 9}, "frequency_penalty"),
        ({"presence_penalty": -9}, "presence_penalty"),
        ({"logit_bias": {"7": 500}}, "logit_bias"),
        ({"logit_bias": {"not a token": 1}}, "logit_bias"),
        ({"logit_bias": [1, 2]}, "logit_bias"),
        ({"n": 0}, "n"),
        ({"n": 2.5}, "n"),
    ],
)
def test_impossible_values_are_refused_with_the_field_named(server, body, expected):
    _, _, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], **body})
    payload = json.loads(response.read())
    connection.close()
    assert response.status == 400
    assert expected in payload["error"]["message"]


def test_n_returns_that_many_choices_each_with_its_own_index(server):
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "n": 3})
    body = json.loads(response.read())
    connection.close()

    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
    assert all(choice["message"]["content"] == "Hello, world" for choice in body["choices"])
    assert len(engine.seen) == 3


def test_n_streams_each_choice_under_its_own_index(server):
    _, _, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "n": 2, "stream": True})
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.read().decode().splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    connection.close()

    indexes = {chunk["choices"][0]["index"] for chunk in chunks}
    assert indexes == {0, 1}
    for index in (0, 1):
        mine = [c for c in chunks if c["choices"][0]["index"] == index]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in mine) == "Hello, world"
        assert mine[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_seeded_request_gives_each_choice_a_different_seed(server):
    """Otherwise `n` returns the same completion `n` times."""
    _, engine, port = server
    connection, response = _post(
        port, {"messages": [{"role": "user", "content": "hi"}], "n": 3, "temperature": 1.0, "seed": 5}
    )
    response.read()
    connection.close()
    assert [seen["sampling"].seed for seen in engine.seen] == [5, 6, 7]


def test_one_choice_is_the_default_and_leaves_the_seed_alone(server):
    _, engine, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "temperature": 1.0, "seed": 5})
    response.read()
    connection.close()
    assert len(engine.seen) == 1
    assert engine.seen[0]["sampling"].seed == 5


def test_a_streamed_seeded_request_also_moves_the_seed_per_choice(server):
    """The streaming path builds its own generations, so it can drop the
    per-choice seed while the whole-response path still carries it."""
    _, engine, port = server
    connection, response = _post(
        port,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "stream": True,
            "temperature": 1.0,
            "seed": 5,
        },
    )
    response.read()
    connection.close()
    assert [seen["sampling"].seed for seen in engine.seen] == [5, 6, 7]


def test_a_streamed_choice_opens_with_a_role_and_closes_with_a_reason(server):
    """What a client needs to start and finish each choice separately."""
    _, _, port = server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "n": 2, "stream": True})
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.read().decode().splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    connection.close()

    assert len({chunk["id"] for chunk in chunks}) == 1
    for index in (0, 1):
        mine = [c for c in chunks if c["choices"][0]["index"] == index]
        assert mine[0]["choices"][0]["delta"]["role"] == "assistant"
        assert mine[-1]["choices"][0]["finish_reason"] == "stop"


class _CrampedEngine(FakeEngine):
    """An engine whose context the request will not fit into."""

    def context_error(self, messages, max_tokens):
        return f"needs more than {self.max_context} tokens"


@pytest.fixture
def cramped_server():
    engine = _CrampedEngine()
    httpd = serve(engine, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, engine, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_a_prompt_over_the_context_is_refused_before_anything_is_sent(cramped_server):
    """By the time the model raises, a streamed response has sent its
    headers and cannot say why it stopped."""
    _, engine, port = cramped_server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}]})
    payload = json.loads(response.read())
    connection.close()

    assert response.status == 400
    assert "8192" in payload["error"]["message"]
    assert engine.seen == []


def test_a_streamed_request_over_the_context_is_refused_too(cramped_server):
    """The streaming path sends its headers first, so the check has to come
    before it, not inside it."""
    _, engine, port = cramped_server
    connection, response = _post(port, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    payload = response.read()
    connection.close()

    assert response.status == 400
    assert response.getheader("Content-Type") == "application/json"
    assert "8192" in json.loads(payload)["error"]["message"]
    assert engine.seen == []
