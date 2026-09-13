"""A thread-safe cache of open file descriptors, one per shard filename.

``os.pread`` never moves the descriptor, so one serves every thread.
"""

import os
import threading
from pathlib import Path


class ShardFdCache:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._fds: dict[str, int] = {}
        # Guards opening only; concurrent `pread`s of an open fd need no lock.
        self._lock = threading.Lock()

    def fd_for_shard(self, shard: str) -> int:
        cached = self._fds.get(shard)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._fds.get(shard)
            if cached is not None:
                return cached
            fd = os.open(str(self.model_dir / shard), os.O_RDONLY)
            self._fds[shard] = fd
            return fd

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
