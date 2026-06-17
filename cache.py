# ============================================================
# cache.py
# ------------------------------------------------------------
# Session cache + pub/sub (local stub for Redis)
#
# What it does:
#   Provides get / set / publish / subscribe over a plain Python
#   dict so live-agent coordination and A/B routing work locally
#   with no Redis running.
#
# How it fits in CallOS:
#   The live voice agent stores per-call state ("call:{sid}:state")
#   and the A/B deployer stores the active adapter here. The
#   interface matches redis-py so production is a drop-in swap.
#
# ADK pattern used:
#   plain module-level helpers (not an ADK construct itself —
#   support code for ADK/Module 7 - Session, State and Memory)
# ============================================================

from collections import defaultdict
from typing import Callable

# TODO: swap to Redis in production (redis.asyncio.Redis with the
# same get/set/publish/subscribe surface).
_STORE: dict[str, str] = {}
_SUBSCRIBERS: dict[str, list[Callable]] = defaultdict(list)


def set(key: str, value: str) -> None:
    """
    Store a value under a key.

    Args:
        key: cache key, e.g. "call:CA123:state"
        value: string value (callers JSON-encode richer data)

    Returns:
        None

    Pattern:
        Mirrors redis.set(). Local impl is a dict write.
    """
    _STORE[key] = value


def get(key: str) -> str | None:
    """
    Read a value, or None if the key is missing.

    Args:
        key: cache key to look up

    Returns:
        str | None — the stored value or None

    Pattern:
        Mirrors redis.get(). Local impl is a dict read.
    """
    return _STORE.get(key)


def publish(channel: str, message: str) -> None:
    """
    Push a message to every subscriber of a channel.

    Args:
        channel: channel name, e.g. "compliance:alerts"
        message: payload string

    Returns:
        None

    Pattern:
        Mirrors redis pub/sub. Local impl calls subscriber callbacks
        synchronously — enough for in-process agent coordination.
    """
    for callback in _SUBSCRIBERS[channel]:
        callback(message)


def subscribe(channel: str, callback: Callable) -> None:
    """
    Register a callback to receive messages on a channel.

    Args:
        channel: channel name to listen on
        callback: function called with each published message

    Returns:
        None

    Pattern:
        Mirrors redis pub/sub subscription registration.
    """
    _SUBSCRIBERS[channel].append(callback)
