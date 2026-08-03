from contextvars import ContextVar
from typing import Any, Optional

search_router: ContextVar[Optional[Any]] = ContextVar("search_router", default=None)
