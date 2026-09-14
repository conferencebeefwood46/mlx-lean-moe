"""The parallel, resumable downloader, against a real local HTTP server
serving real byte ranges rather than a mock.

The failure these exist for: a retry re-appending bytes it already wrote,
leaving a part whose content is wrong at a plausible size. So: bytes.
"""

import hashlib
import http.server
import os
import threading
from functools import partial
from pathlib import Path

import httpx
import pytest

from mlx_lean_moe.weights.download import (
    RemoteFile,
    _download_file,
    _fetch_range,
    blob_id,
    download_repo,
    is_complete,
    repo_dir,
    sha256_of,
    snapshot_dir,
)

COMMIT = "0" * 40
PAYLOAD = bytes((i * 7 + 11) % 256 for i in range(300_000))


SERVED = {"model.bin": PAYLOAD, "small.json": b'{"hello": "world"}'}


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves byte ranges properly; `SimpleHTTPRequestHandler` ignores the
    Range header, which would pass every range test for the wrong reason."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        data = SERVED.get(self.path.lstrip("/"))
        if data is None:
            self.send_error(404)
            return

        header = self.headers.get("Range")
        if header:
            first, _, last = header.removeprefix("bytes=").partition("-")
            start = int(first)
            end = int(last) if last else len(data) - 1
            if start >= len(data):
                self.send_error(416)
                return
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep test output readable


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


class _IgnoresRangeHandler(http.server.BaseHTTPRequestHandler):
    """Answers every request with the whole file, Range header or not, as
    some proxies do."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server_ignoring_ranges():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _IgnoresRangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def client():
    with httpx.Client(follow_redirects=True) as c:
        yield c


def test_fetch_range_downloads_exactly_the_requested_bytes(server, client, tmp_path):
    out = tmp_path / "part"
    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999)
    assert out.read_bytes() == PAYLOAD[1000:2000]


def test_fetch_range_resumes_from_a_partial_part(server, client, tmp_path):
    """Simulates an interrupted transfer: a part holding a valid prefix must
    be continued, not restarted and not appended to blindly."""
    out = tmp_path / "part"
    out.write_bytes(PAYLOAD[1000:1400])  # 400 of the 1000 bytes we want

    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999)

    assert out.read_bytes() == PAYLOAD[1000:2000]


def test_fetch_range_restarts_an_oversized_part(server, client, tmp_path):
    """A part longer than its range holds duplicated bytes interleaved, so
    it cannot be truncated back into shape."""
    out = tmp_path / "part"
    out.write_bytes(PAYLOAD[1000:1400] + PAYLOAD[1000:2000])  # duplicated prefix

    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999)

    assert out.read_bytes() == PAYLOAD[1000:2000]


def test_fetch_range_reports_a_resumed_prefix_without_refetching_it(
    server, client, tmp_path
):
    """Bytes already on disk are progress too; ignoring them starts a
    resumed download at 0%."""
    out = tmp_path / "part"
    out.write_bytes(PAYLOAD[1000:1400])
    deltas = []

    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999, on_bytes=deltas.append)

    assert deltas[0] == 400  # what was already there, reported before any fetch
    assert sum(deltas) == 1000


def test_fetch_range_starting_from_an_oversized_part_still_totals_the_range(
    server, client, tmp_path
):
    """After a discard and refetch the deltas must still add up to the range
    itself, not to the range plus what the bad file held."""
    out = tmp_path / "part"
    out.write_bytes(
        PAYLOAD[1000:1400] + PAYLOAD[1000:2000]
    )  # 1400 bytes for a 1000-byte range
    deltas = []

    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999, on_bytes=deltas.append)

    assert sum(deltas) == 1000


def test_fetch_range_takes_back_the_bytes_of_a_part_it_discards(
    server_ignoring_ranges, tmp_path
):
    """Bytes reported for a part that is then discarded have to be taken
    back, or every doomed attempt drives the total past 100%."""
    out = tmp_path / "part"
    deltas = []

    with httpx.Client(follow_redirects=True) as client, pytest.raises(RuntimeError):
        _fetch_range(
            client,
            f"{server_ignoring_ranges}/model.bin",
            out,
            1000,
            1999,
            attempts=3,
            on_bytes=deltas.append,
        )

    assert any(d < 0 for d in deltas), (
        "the discarded part's bytes were never taken back"
    )
    # Three attempts each fetched the whole 300kB payload for a 1000-byte
    # range, and the total still reflects only what is on disk at the end.
    assert sum(deltas) == (out.stat().st_size if out.exists() else 0)


def test_fetch_range_is_a_no_op_when_the_part_is_already_complete(
    server, client, tmp_path
):
    out = tmp_path / "part"
    out.write_bytes(PAYLOAD[1000:2000])
    mtime = out.stat().st_mtime_ns

    _fetch_range(client, f"{server}/model.bin", out, 1000, 1999)

    assert out.stat().st_mtime_ns == mtime  # untouched


def test_fetch_range_gives_up_with_a_clear_error(client, tmp_path):
    out = tmp_path / "part"
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        _fetch_range(client, "http://127.0.0.1:1/nope", out, 0, 99, attempts=2)


def test_download_file_assembles_parts_in_order(server, client, tmp_path):
    target = tmp_path / "model.bin"
    remote = RemoteFile("model.bin", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())

    _download_file(
        client, f"{server}/model.bin", target, remote, chunk_size=32_000, connections=4
    )

    assert target.read_bytes() == PAYLOAD
    # Parts are cleaned up once the whole file verifies.
    assert not list((tmp_path / ".parts").glob("model.bin.*"))


def test_download_file_rejects_a_wrong_sha256(server, client, tmp_path):
    target = tmp_path / "model.bin"
    remote = RemoteFile("model.bin", len(PAYLOAD), "0" * 64)

    with pytest.raises(RuntimeError, match="sha256"):
        _download_file(
            client,
            f"{server}/model.bin",
            target,
            remote,
            chunk_size=32_000,
            connections=4,
        )

    # The bad assembly is removed rather than left to be loaded later.
    assert not target.exists()


def test_download_file_skips_a_file_that_is_already_complete(server, client, tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(PAYLOAD)
    mtime = target.stat().st_mtime_ns
    remote = RemoteFile("model.bin", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())

    _download_file(
        client, f"{server}/model.bin", target, remote, chunk_size=32_000, connections=4
    )

    assert target.stat().st_mtime_ns == mtime


def test_download_file_handles_a_file_smaller_than_one_chunk(server, client, tmp_path):
    target = tmp_path / "small.json"
    remote = RemoteFile("small.json", 18, None)

    _download_file(
        client, f"{server}/small.json", target, remote, chunk_size=32_000, connections=4
    )

    assert target.read_bytes() == b'{"hello": "world"}'


def test_download_file_reports_every_byte_exactly_once(server, client, tmp_path):
    """The deltas have to sum to the file, not to some multiple of it: a
    retry re-reporting a range it already counted inflates the total."""
    deltas = []
    remote = RemoteFile("model.bin", len(PAYLOAD), None)

    _download_file(
        client,
        f"{server}/model.bin",
        tmp_path / "model.bin",
        remote,
        chunk_size=100_000,
        connections=2,
        on_bytes=deltas.append,
    )

    assert sum(deltas) == len(PAYLOAD)
    assert len(deltas) > 3  # per network block, not per 100_000-byte range


def _serve_locally(monkeypatch, server):
    """Points the URLs `download_repo` builds at the local test server, so
    these need no network metadata call."""
    import mlx_lean_moe.weights.download as dl

    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.httpx.Client", partial(httpx.Client, base_url="")
    )
    original = dl._download_file

    def to_local(client, url, target, remote, *args, **kwargs):
        return original(
            client, f"{server}/{remote.name}", target, remote, *args, **kwargs
        )

    monkeypatch.setattr(dl, "_download_file", to_local)


def test_download_repo_writes_every_file_and_resumes(monkeypatch, server, tmp_path):
    """End-to-end through the public entry point, with the file list
    injected so the test needs no network metadata call."""
    files = [
        RemoteFile("small.json", 18, None),
        RemoteFile("model.bin", len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest()),
    ]
    _serve_locally(monkeypatch, server)

    dest = download_repo(
        "some-org/some-model",
        root=tmp_path,
        files=files,
        chunk_size=64_000,
        connections=3,
        commit=COMMIT,
    )

    assert dest == snapshot_dir("some-org/some-model", tmp_path)
    assert (dest / "model.bin").read_bytes() == PAYLOAD
    assert (dest / "small.json").read_bytes() == b'{"hello": "world"}'
    assert not (
        repo_dir("some-org/some-model", tmp_path) / ".parts"
    ).exists()  # cleaned up when empty


def test_download_repo_reports_progress_across_the_whole_repo(
    monkeypatch, server, tmp_path
):
    """Totals are repo-wide: per file, a bar restarts at 0% once per shard.
    The last call reports the whole thing done."""
    files = [
        RemoteFile("small.json", 18, None),
        RemoteFile("model.bin", len(PAYLOAD), None),
    ]
    _serve_locally(monkeypatch, server)
    seen = []

    download_repo(
        "some-org/some-model",
        root=tmp_path,
        files=files,
        chunk_size=64_000,
        connections=3,
        commit=COMMIT,
        on_progress=lambda name, done, total: seen.append((name, done, total)),
    )

    whole = 18 + len(PAYLOAD)
    assert {total for _, _, total in seen} == {whole}
    assert seen[-1][1] == whole
    assert {name for name, _, _ in seen} == {"small.json", "model.bin"}


def test_download_repo_counts_files_that_were_already_there(
    monkeypatch, server, tmp_path
):
    """Resuming a checkpoint whose small files landed before the interruption
    must not start the bar at 0%: those bytes are on disk and are progress."""
    files = [
        RemoteFile("small.json", 18, None),
        RemoteFile("model.bin", len(PAYLOAD), None),
    ]
    dest = repo_dir("some-org/some-model", tmp_path) / "snapshots" / COMMIT
    dest.mkdir(parents=True)
    (dest / "small.json").write_bytes(b'{"hello": "world"}')
    _serve_locally(monkeypatch, server)
    seen = []

    download_repo(
        "some-org/some-model",
        root=tmp_path,
        files=files,
        chunk_size=64_000,
        connections=3,
        commit=COMMIT,
        on_progress=lambda name, done, total: seen.append((name, done, total)),
    )

    assert seen[0][1] >= 18  # the already-present file, before a byte was fetched
    assert seen[-1][1] == 18 + len(PAYLOAD)


def test_the_cache_layout_is_the_hub_s_own(tmp_path):
    """`models--owner--name`, reached through `refs` and `snapshots`: what
    every other tool reading this cache expects."""
    assert repo_dir("org/name", tmp_path) == tmp_path / "models--org--name"
    assert snapshot_dir("org/name", tmp_path) is None  # nothing cached yet

    repo = repo_dir("org/name", tmp_path)
    (repo / "snapshots" / COMMIT).mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(
        COMMIT + "\n"
    )  # git writes a newline; resolving must not care
    assert snapshot_dir("org/name", tmp_path) == repo / "snapshots" / COMMIT
    # A commit is taken as one, without consulting refs.
    assert snapshot_dir("org/name", tmp_path, COMMIT) == repo / "snapshots" / COMMIT
    assert snapshot_dir("org/name", tmp_path, "other-branch") is None


def test_a_blob_is_named_the_way_the_hub_names_it(tmp_path):
    """An LFS file by its published sha256, everything else by its git blob
    hash, or the hub's own tools fetch it again."""
    path = tmp_path / "small.json"
    path.write_bytes(b"hello")

    assert (
        blob_id(RemoteFile("small.json", 5, "published-sha256"), path)
        == "published-sha256"
    )
    # git hash-object: sha1 over "blob <size>\0" then the bytes.
    assert (
        blob_id(RemoteFile("small.json", 5, None), path)
        == hashlib.sha1(b"blob 5\0hello").hexdigest()
    )


def test_a_downloaded_file_is_a_link_into_blobs(monkeypatch, server, tmp_path):
    """The bytes live once, under their hash, and the snapshot points at
    them. Relatively, so moving the cache does not break every link."""
    payload_sha = hashlib.sha256(PAYLOAD).hexdigest()
    files = [RemoteFile("model.bin", len(PAYLOAD), payload_sha)]
    _serve_locally(monkeypatch, server)

    dest = download_repo(
        "some-org/some-model",
        root=tmp_path,
        files=files,
        chunk_size=64_000,
        connections=3,
        commit=COMMIT,
    )

    link = dest / "model.bin"
    assert link.is_symlink()
    assert not os.path.isabs(os.readlink(link))
    assert (
        link.resolve()
        == (repo_dir("some-org/some-model", tmp_path) / "blobs" / payload_sha).resolve()
    )
    assert (
        repo_dir("some-org/some-model", tmp_path) / "refs" / "main"
    ).read_text().strip() == COMMIT


def test_sha256_of_matches_hashlib(tmp_path):
    path = Path(tmp_path / "blob")
    path.write_bytes(PAYLOAD)
    assert sha256_of(path, block_size=1024) == hashlib.sha256(PAYLOAD).hexdigest()


# `_exploding_repo_files` makes "decides from local bytes" testable: any
# network call on the happy path fails the test.


def _write_runnable_checkpoint(directory: Path) -> None:
    """The smallest directory this loader would accept: a config, a
    tokenizer, and one whole shard."""
    import numpy as np
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type": "fake"}')
    (directory / "tokenizer.json").write_text("{}")
    save_file(
        {"a": np.zeros(8, dtype=np.float32)}, str(directory / "model.safetensors")
    )


def _exploding_repo_files(*args, **kwargs):
    raise AssertionError(
        "is_complete asked the hub about an already-complete checkpoint"
    )


def _cached_snapshot(repo_id: str, root) -> "Path":
    """A repo already in the cache: a snapshot with `refs/main` pointing at
    it, which is what `is_complete` has to look through."""
    repo = repo_dir(repo_id, root)
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(COMMIT)
    return snapshot


def test_a_complete_checkpoint_never_asks_the_hub(tmp_path, monkeypatch):
    _write_runnable_checkpoint(_cached_snapshot("org/model", tmp_path))
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.repo_files", _exploding_repo_files
    )
    assert is_complete("org/model", root=tmp_path) is True


def test_a_truncated_shard_falls_back_to_the_hub(tmp_path, monkeypatch):
    """Local bytes say "not all here", and only the hub can say whether
    that's the whole story -- so this is where the network call belongs."""
    directory = _cached_snapshot("org/model", tmp_path)
    _write_runnable_checkpoint(directory)
    shard = directory / "model.safetensors"
    with shard.open("r+b") as f:
        f.truncate(shard.stat().st_size - 4)

    asked = []

    def fake_repo_files(repo_id, revision=None):
        asked.append(repo_id)
        return [RemoteFile("model.safetensors", shard.stat().st_size, None)]

    monkeypatch.setattr("mlx_lean_moe.weights.download.repo_files", fake_repo_files)
    # The hub is consulted, and agrees the file is the size it now is --
    # this pins that the fallback path is reached, not its verdict.
    assert is_complete("org/model", root=tmp_path) is True
    assert asked == ["org/model"]


def test_missing_tokenizer_is_incomplete_even_with_whole_shards(tmp_path, monkeypatch):
    """Complete weights do not imply a runnable directory, however the
    directory was assembled."""
    directory = _cached_snapshot("org/model", tmp_path)
    _write_runnable_checkpoint(directory)
    (directory / "tokenizer.json").unlink()
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.repo_files", _exploding_repo_files
    )
    assert is_complete("org/model", root=tmp_path) is False


def test_leftover_parts_mean_a_download_stopped_partway(tmp_path, monkeypatch):
    """Shards can be whole while a later file is still mid-flight."""
    directory = _cached_snapshot("org/model", tmp_path)
    _write_runnable_checkpoint(directory)
    parts = repo_dir("org/model", tmp_path) / ".parts"
    parts.mkdir()
    (parts / "tokenizer.json.0000").write_bytes(b"partial")
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.repo_files", _exploding_repo_files
    )
    assert is_complete("org/model", root=tmp_path) is False


def test_a_directory_that_was_never_downloaded_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.repo_files", _exploding_repo_files
    )
    assert is_complete("org/nothing", root=tmp_path) is False


def test_a_rate_limited_range_waits_before_retrying(tmp_path, monkeypatch):
    """429 means "not now", not "no"; retrying at once with every range in
    flight spends the attempt budget in seconds."""
    slept: list[float] = []
    monkeypatch.setattr("mlx_lean_moe.weights.download.time.sleep", slept.append)

    attempts = {"n": 0}

    class _Refusing:
        def stream(self, method, url, headers=None):
            attempts["n"] += 1
            raise httpx.HTTPStatusError(
                "too many requests",
                request=httpx.Request("GET", url),
                # Named a wait, so that is what must be honoured.
                response=httpx.Response(429, headers={"retry-after": "7"}),
            )

    with pytest.raises(RuntimeError):
        _fetch_range(
            _Refusing(), "http://example/x", tmp_path / "part", 0, 99, attempts=3
        )

    assert attempts["n"] == 3
    assert slept == [7.0, 7.0, 7.0]


def test_backoff_grows_when_the_server_names_no_delay(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("mlx_lean_moe.weights.download.time.sleep", slept.append)

    class _Overloaded:
        def stream(self, method, url, headers=None):
            raise httpx.HTTPStatusError(
                "unavailable",
                request=httpx.Request("GET", url),
                response=httpx.Response(503),
            )

    with pytest.raises(RuntimeError):
        _fetch_range(
            _Overloaded(), "http://example/x", tmp_path / "part", 0, 99, attempts=4
        )

    assert slept == [1.0, 2.0, 4.0, 8.0]


def test_an_ordinary_failure_is_retried_without_waiting(tmp_path, monkeypatch):
    """A dropped connection is not a refusal: whatever landed is a valid
    prefix and the next pass resumes from it, so waiting only wastes time."""
    slept: list[float] = []
    monkeypatch.setattr("mlx_lean_moe.weights.download.time.sleep", slept.append)

    class _Dropping:
        def stream(self, method, url, headers=None):
            raise httpx.ConnectError("dropped")

    with pytest.raises(RuntimeError):
        _fetch_range(
            _Dropping(), "http://example/x", tmp_path / "part", 0, 99, attempts=3
        )

    assert slept == []
