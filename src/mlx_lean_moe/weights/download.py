"""Parallel, resumable checkpoint downloads into the Hugging Face cache.

Each file is split into byte ranges fetched concurrently, every range
resumes where it stopped, and the assembled file is checked against the
repo's published sha256. The cache layout is the hub's own -- blobs named by
etag, a symlink per file under snapshots, the commit in refs -- so every
other tool reading it finds the same copy.
"""

import hashlib
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi

from mlx_lean_moe.weights import safetensors_index

DEFAULT_CHUNK_SIZE = 48 * 1024 * 1024
DEFAULT_CONNECTIONS = 24
_MIN_NAME_ROOM = 20
DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=180.0)


@dataclass(frozen=True, slots=True)
class RemoteFile:
    name: str
    size: int
    sha256: str | None  # published for LFS files (the large ones) only


def _hub_root(root: Path | None = None) -> Path:
    """The hub cache directory, as `huggingface_hub` itself resolves it."""

    if root is not None:
        return Path(root)

    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def repo_dir(repo_id: str, root: Path | None = None) -> Path:
    """A repo's cache entry: `models--<owner>--<name>`."""
    return _hub_root(root) / ("models--" + repo_id.replace("/", "--"))


def snapshot_dir(
    repo_id: str, root: Path | None = None, revision: str | None = None
) -> Path | None:
    """The directory of files for a revision, or None when it is not cached."""

    repo = repo_dir(repo_id, root)
    commit = revision or "main"

    if not _is_commit(commit):
        reference = repo / "refs" / commit
        if not reference.exists():
            return None

        commit = reference.read_text().strip()

    snapshot = repo / "snapshots" / commit

    return snapshot if snapshot.is_dir() else None


def _is_commit(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def _repo_revision(repo_id: str, revision: str | None = None) -> str:
    """The commit a revision resolves to."""
    return HfApi().model_info(repo_id, revision=revision).sha


def blob_id(remote: RemoteFile, path: Path) -> str:
    """What the hub names a blob: an LFS file's published sha256, and for
    anything else the git blob sha1 over `blob <size>\0` and the bytes."""

    if remote.sha256:
        return remote.sha256

    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size, usedforsecurity=False)

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def repo_files(repo_id: str, revision: str | None = None) -> list[RemoteFile]:
    """Every file in the repo, smallest first so a failure surfaces before
    hours of shard downloading rather than after."""

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = [
        RemoteFile(
            s.rfilename, s.size, getattr(getattr(s, "lfs", None), "sha256", None)
        )
        for s in info.siblings
        if s.size is not None and not s.rfilename.startswith(".")
    ]

    return sorted(files, key=lambda f: f.size)


def sha256_of(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)

    return digest.hexdigest()


# Statuses that mean "not now" rather than "no".
_BACKS_OFF = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF = 30.0


def _retry_after(response) -> float:
    """Seconds the server asked us to wait, if it named a number."""

    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return 0.0


def _fetch_range(
    client: httpx.Client,
    url: str,
    out: Path,
    start: int,
    end: int,
    attempts: int = 40,
    on_bytes: Callable[[int], None] | None = None,
) -> None:
    """Fetch ``[start, end]`` into ``out``, resuming from the file's length
    each attempt. ``on_bytes`` gets the change on disk, negatives too."""

    want = end - start + 1
    reported = 0

    def report(now: int) -> None:
        nonlocal reported

        if on_bytes is not None and now != reported:
            on_bytes(now - reported)

        reported = now

    last_error: Exception | None = None
    refusals = 0

    for _ in range(attempts):
        have = out.stat().st_size if out.exists() else 0
        if have == want:
            report(have)

            return

        if have > want:
            # Duplicated bytes sit interleaved, not as a clean prefix.
            out.unlink()

            have = 0

        report(have)

        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()

                written = 0

                with open(out, "ab") as fh:
                    for block in response.iter_bytes():
                        fh.write(block)

                        written += len(block)
                        report(have + written)
        except httpx.HTTPStatusError as exc:
            last_error = exc

            if exc.response.status_code not in _BACKS_OFF:
                continue

            # Refused concurrency, not a failure: retrying at once turns a
            # brief limit into a long one.
            wait = _retry_after(exc.response) or min(_MAX_BACKOFF, 2.0**refusals)
            refusals += 1

            time.sleep(wait)
        except (httpx.HTTPError, OSError) as exc:
            # A truncated stream is fine -- whatever landed is a valid
            # prefix, and the next pass picks up from there.
            last_error = exc

    have = out.stat().st_size if out.exists() else 0
    raise RuntimeError(
        f"{out.name}: got {have}/{want} bytes after {attempts} attempts (last error: {last_error})"
    )


def _download_file(
    client: httpx.Client,
    url: str,
    target: Path,
    remote: RemoteFile,
    chunk_size: int,
    connections: int,
    on_bytes: Callable[[int], None] | None = None,
) -> None:
    if target.exists() and target.stat().st_size == remote.size:
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    ranges = [
        (start, min(start + chunk_size - 1, remote.size - 1))
        for start in range(0, max(remote.size, 1), chunk_size)
    ]

    if len(ranges) == 1:
        _fetch_range(client, url, target, 0, remote.size - 1, on_bytes=on_bytes)

        parts: list[Path] = []
    else:
        parts = _fetch_in_parts(
            client, url, target, remote, ranges, connections, on_bytes
        )

    _verify(target, remote)

    for part in parts:
        part.unlink(missing_ok=True)


def _fetch_in_parts(
    client: httpx.Client,
    url: str,
    target: Path,
    remote: RemoteFile,
    ranges: list[tuple[int, int]],
    connections: int,
    on_bytes: Callable[[int], None] | None,
) -> list[Path]:
    parts_dir = target.parent / ".parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        parts_dir / f"{Path(remote.name).name}.{i:04d}" for i in range(len(ranges))
    ]

    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [
            pool.submit(_fetch_range, client, url, part, start, end, on_bytes=on_bytes)
            for part, (start, end) in zip(parts, ranges)
        ]
        for future in as_completed(futures):
            future.result()

    for part, (start, end) in zip(parts, ranges):
        actual = part.stat().st_size
        if actual != end - start + 1:
            raise RuntimeError(
                f"{part.name}: {actual} bytes, expected {end - start + 1}"
            )

    with open(target, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())

    return parts


def _verify(target: Path, remote: RemoteFile) -> None:
    size = target.stat().st_size
    if size != remote.size:
        raise RuntimeError(
            f"{remote.name}: assembled {size} bytes, expected {remote.size}"
        )

    # Size alone passed for parts that held the wrong content at a plausible
    # length, so the published hash is checked whenever there is one.
    if remote.sha256:
        digest = sha256_of(target)
        if digest != remote.sha256:
            target.unlink()

            raise RuntimeError(
                f"{remote.name}: sha256 {digest[:12]}... != published {remote.sha256[:12]}..."
            )


def _link(snapshot: Path, name: str, blob: Path) -> None:
    """Point `snapshot/name` at its blob, relatively, so the cache survives
    being moved or mounted somewhere else."""

    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink() or target.exists():
        target.unlink()

    target.symlink_to(os.path.relpath(blob, target.parent))


def download_repo(
    repo_id: str,
    root: Path | None = None,
    revision: str | None = None,
    connections: int = DEFAULT_CONNECTIONS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_progress: Callable[[str, int, int], None] | None = None,
    files: Iterable[RemoteFile] | None = None,
    commit: str | None = None,
) -> Path:
    """Download ``repo_id`` into the hub cache, returning the snapshot dir.
    ``on_progress(file, done, total)`` gets repo-wide totals under a lock."""

    remote_files = list(files) if files is not None else repo_files(repo_id, revision)
    commit = commit or _repo_revision(repo_id, revision)

    repo = repo_dir(repo_id, root)
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / commit

    blobs.mkdir(parents=True, exist_ok=True)

    snapshot.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(f.size for f in remote_files)
    done_bytes = sum(
        f.size for f in remote_files if _already_whole(snapshot / f.name, f.size)
    )
    lock = threading.Lock()
    current = ""

    def bump(delta: int) -> None:
        nonlocal done_bytes

        with lock:
            done_bytes += delta
            on_progress(current, done_bytes, total_bytes)

    ref = revision or "main"
    with httpx.Client(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
        for remote in remote_files:
            current = remote.name

            if _already_whole(snapshot / remote.name, remote.size):
                continue

            # An LFS blob's name is known up front; anything else is hashed
            # once whole, so it lands under a temporary name first.
            settled = blobs / remote.sha256 if remote.sha256 else None
            target = settled or (
                repo / ".parts" / f"{remote.name.replace('/', '--')}.incoming"
            )
            target.parent.mkdir(parents=True, exist_ok=True)

            _download_file(
                client,
                f"https://huggingface.co/{repo_id}/resolve/{ref}/{remote.name}",
                target,
                remote,
                chunk_size,
                connections,
                bump if on_progress else None,
            )

            if settled is None:
                settled = blobs / blob_id(remote, target)
                target.replace(settled)

            _link(snapshot, remote.name, settled)

    reference = repo / "refs" / ref
    reference.parent.mkdir(parents=True, exist_ok=True)

    reference.write_text(commit)

    parts_dir = repo / ".parts"
    if parts_dir.exists() and not any(parts_dir.iterdir()):
        shutil.rmtree(parts_dir)

    return snapshot


def _already_whole(path: Path, size: int) -> bool:
    """A symlink into blobs counts, which is what resuming a finished file
    looks like."""

    try:
        return path.stat().st_size == size
    except OSError:
        return False


def is_complete(
    repo_id: str, root: Path | None = None, revision: str | None = None
) -> bool:
    """Whether the repo is cached and whole, answered from local bytes since
    this runs on every startup; the hub is asked only if they fall short."""

    snapshot = snapshot_dir(repo_id, root, revision)
    if snapshot is None:
        return False

    if not (snapshot / "config.json").exists():
        return False

    if not any(
        (snapshot / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        return False

    parts_dir = repo_dir(repo_id, root) / ".parts"
    if parts_dir.exists() and any(parts_dir.iterdir()):
        return False

    if safetensors_index.checkpoint_is_complete(snapshot):
        return True

    try:
        remote_files = repo_files(repo_id, revision)
    except Exception:
        return False

    return all(_already_whole(snapshot / f.name, f.size) for f in remote_files)


def cached_checkpoints(root: Path | None = None) -> list[str]:
    """Every model repo in the cache that is downloaded and whole."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir(_hub_root(root))
    except Exception:
        return []

    repos = (repo.repo_id for repo in cache.repos if repo.repo_type == "model")

    return sorted(repo_id for repo_id in repos if is_complete(repo_id, root))


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if num_bytes < 1024 or unit == "T":
            return f"{num_bytes:.1f}{unit}"

        num_bytes /= 1024.0

    return f"{num_bytes:.1f}T"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"

    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"

    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


class DownloadProgress:
    """One live progress line for a whole download, called once per network
    block by the downloading threads, hence the `_MIN_INTERVAL` gate."""

    _MIN_INTERVAL = 0.1  # seconds between redraws
    _WINDOW = 20.0  # seconds of history behind the rate estimate
    _LOG_STEP = 0.05  # fraction between lines when not a terminal

    def __init__(self, stream=None, now: Callable[[], float] = time.monotonic) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._now = now
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._samples: deque[tuple[float, int]] = deque()
        self._last_draw = -math.inf  # so the first update draws at once
        self._last_logged = 0.0
        self._width = 0

    def update(self, name: str, done: int, total: int) -> None:
        now = self._now()
        self._samples.append((now, done))

        while len(self._samples) > 2 and now - self._samples[0][0] > self._WINDOW:
            self._samples.popleft()

        if not self._tty:
            fraction = done / total if total else 1.0
            if fraction - self._last_logged >= self._LOG_STEP or done >= total:
                self._last_logged = fraction
                print(
                    f"  {fraction:5.1%}  {_human_size(done)}/{_human_size(total)}  {name}",
                    file=self._stream,
                )

            return

        if now - self._last_draw < self._MIN_INTERVAL and done < total:
            return

        self._last_draw = now
        self._draw(name, done, total)

    def _rate(self) -> float:
        """Bytes per second over the window, or 0 with too little history."""

        if len(self._samples) < 2:
            return 0.0

        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]

        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    _BAR_WIDTH = 26

    def _draw(self, name: str, done: int, total: int) -> None:
        fraction = done / total if total else 1.0
        rate = self._rate()
        eta = (
            f"eta {_duration((total - done) / rate)}"
            if rate > 0 and done < total
            else "eta --"
        )
        # Padded to the widest form of each field: nothing left of the
        # filename shifts as the numbers change.
        stats = f"{fraction:6.1%} {_human_size(done):>6}/{_human_size(total):<6} {rate / 2**20:5.1f} MB/s {eta:<9}"

        # The bar is the first thing to go on a narrow terminal.
        columns = shutil.get_terminal_size((80, 24)).columns
        if columns >= self._BAR_WIDTH + len(stats) + _MIN_NAME_ROOM:
            filled = round(self._BAR_WIDTH * fraction)
            line = f"[{'#' * filled}{'-' * (self._BAR_WIDTH - filled)}] {stats}  {name}"
        else:
            line = f"{stats}  {name}"

        line = line[: columns - 1]
        self._stream.write("\r" + line + " " * max(0, self._width - len(line)))

        self._stream.flush()

        self._width = len(line)

    def close(self) -> None:
        """Leave the finished line in place and move off it."""

        if self._tty and self._width:
            self._stream.write("\n")

            self._stream.flush()

            self._width = 0


def fetch(
    repo_id: str,
    revision: str | None = None,
    *,
    root: Path | None = None,
    connections: int = DEFAULT_CONNECTIONS,
    stream=None,
) -> Path:
    """Make sure `repo_id` is in the cache and return its directory,
    downloading it with a progress line when it is not."""

    if is_complete(repo_id, root, revision):
        cached = snapshot_dir(repo_id, root, revision)
        if cached is not None:
            return cached

    bar = DownloadProgress(stream)
    try:
        return download_repo(
            repo_id,
            root=root,
            revision=revision,
            connections=connections,
            on_progress=bar.update,
        )
    finally:
        bar.close()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a checkpoint from the Hugging Face hub into its cache, over many connections at once."
    )
    parser.add_argument(
        "repo_id", help="for example froggeric/Qwen3.6-35B-A3B-...-MLX-4bit"
    )

    parser.add_argument(
        "--revision", default=None, help="branch or commit (default: main)"
    )

    parser.add_argument(
        "--connections",
        type=int,
        default=DEFAULT_CONNECTIONS,
        help=f"default {DEFAULT_CONNECTIONS}",
    )

    arguments = parser.parse_args()

    path = fetch(
        arguments.repo_id, arguments.revision, connections=arguments.connections
    )
    print(path)


if __name__ == "__main__":
    _main()
