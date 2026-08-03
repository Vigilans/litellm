"""
Internal request context for LiteLLM.

Provides a ContextVar-based mechanism for internal signals that must not
be settable from user input. Context variables are scoped to the current
asyncio task and cannot be injected via HTTP request bodies.
"""

from contextvars import ContextVar

# When True, suppresses async logging and billing for internal sub-calls
# (e.g., emulated file-search steps that make nested LLM calls).
is_internal_call: ContextVar[bool] = ContextVar("is_internal_call", default=False)

# When True, the request is a web search LiteLLM is running on a caller's
# behalf. Web search interception rewrites the search tool such a request
# carries, which would make the search search again.
is_web_search_call: ContextVar[bool] = ContextVar("is_web_search_call", default=False)
