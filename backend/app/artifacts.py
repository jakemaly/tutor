"""Short-lived in-memory store for generated files, so bytes never pass through the model."""

import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

DEFAULT_TTL_SECONDS = 30 * 60
DEFAULT_MAX_ITEMS = 50
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    id: str
    filename: str
    media_type: str
    data: bytes
    expires_at: float


class ArtifactBackend(Protocol):
    """What the tool and the download route need from a store.

    ArtifactStore below keeps files in this process's memory. A deployment that spans several
    processes or serverless instances should pass an implementation backed by shared storage
    (for example object storage with signed URLs) to create_app(artifacts=...).
    """

    def put(self, filename: str, media_type: str, data: bytes) -> Artifact: ...

    def get(self, artifact_id: str) -> Artifact | None: ...


class ArtifactStore:
    """Capability-URL storage: an unguessable id is the only key, and entries expire quickly.

    State is per process. A multi-instance or serverless deployment needs a shared store
    (for example object storage with signed URLs) behind this same put/get interface.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ttl, self._max_items, self._max_bytes, self._clock = (
            ttl_seconds,
            max_items,
            max_bytes,
            clock,
        )
        self._items: OrderedDict[str, Artifact] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, filename: str, media_type: str, data: bytes) -> Artifact:
        if len(data) > self._max_bytes:
            raise ValueError("File is too large to store.")
        artifact = Artifact(
            secrets.token_urlsafe(24), filename, media_type, data, self._clock() + self._ttl
        )
        with self._lock:
            self._purge_expired()
            self._items[artifact.id] = artifact
            # Oldest first: stay inside both the item and total-size budgets.
            while len(self._items) > self._max_items or self._total_bytes() > self._max_bytes:
                self._items.popitem(last=False)
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        with self._lock:
            self._purge_expired()
            return self._items.get(artifact_id)

    def _purge_expired(self) -> None:
        now = self._clock()
        for key in [k for k, item in self._items.items() if item.expires_at <= now]:
            del self._items[key]

    def _total_bytes(self) -> int:
        return sum(len(item.data) for item in self._items.values())
