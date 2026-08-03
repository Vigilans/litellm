from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.responses.streaming_iterator import ResponsesAPIEventStreamIterator
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


async def event_stream(*events):
    for event in events:
        yield event


def logging_obj():
    logger = MagicMock()
    logger.async_success_handler = AsyncMock()
    logger.async_failure_handler = AsyncMock()
    logger.start_time = datetime.now()
    logger.model_call_details = {"litellm_params": {}}
    return logger


@pytest.mark.asyncio
async def test_completed_event_is_recorded_and_logged_once():
    response = ResponsesAPIResponse(
        id="resp_1",
        created_at=1,
        model="claude-opus-5",
        object="response",
        output=[],
        status="completed",
    )
    completed = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
        sequence_number=1,
    )
    logger = logging_obj()
    iterator = ResponsesAPIEventStreamIterator(
        event_iterator=event_stream(completed),
        model="claude-opus-5",
        logging_obj=logger,
        call_type="aresponses",
    )

    assert [event async for event in iterator] == [completed]
    assert iterator.completed_response is completed
    logger.async_success_handler.assert_called_once()
    logger.success_handler.assert_called_once()


@pytest.mark.asyncio
async def test_error_event_uses_failure_logging():
    error = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=1,
        error=ErrorEventError(type="api_error", code="api_error", message="failed"),
    )
    logger = logging_obj()
    iterator = ResponsesAPIEventStreamIterator(
        event_iterator=event_stream(error),
        model="claude-opus-5",
        logging_obj=logger,
    )

    assert [event async for event in iterator] == [error]
    logger.async_failure_handler.assert_called_once()
    logger.async_success_handler.assert_not_called()
