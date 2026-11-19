"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self.last_blocked = False  # did the most recent request hit the limit?

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Slide the window: forget requests older than window_seconds
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Over the limit → block without calling the LLM.
        #    Blocked requests are NOT recorded, so a flooding user is not locked out forever.
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self.last_blocked = True
            return self._block_response(
                f"Rate limit exceeded: max {self.max_requests} requests per "
                f"{self.window_seconds}s. Try again in {max(wait, 0):.0f}s."
            )

        # 3. Under the limit → record and let it through
        window.append(now)
        self.last_blocked = False
        return None

    def reset(self):
        """Clear all windows and counters (used between test groups)."""
        self.user_windows.clear()
        self.blocked_count = 0
        self.total_count = 0
        self.last_blocked = False
