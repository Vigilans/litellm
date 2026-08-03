"""
Routes a Responses API request to a provider's native Anthropic ``/v1/messages``
endpoint, translating in both directions.
"""

import time
from typing import Any, AsyncIterator, Coroutine, Dict, Optional, Union, cast

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)

from .streaming_iterator import AnthropicMessagesToResponsesStreamIterator
from .transformation import (
    translate_anthropic_messages_response_to_responses_api_response,
    translate_responses_request_to_anthropic_messages_request,
)


class LiteLLMResponsesToMessagesAPIHandler:
    """
    Bridges ``/v1/responses`` to providers that serve Claude models through their
    native ``/v1/messages`` endpoint.
    """

    @staticmethod
    def response_api_handler(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        AsyncIterator[Any],
        Coroutine[Any, Any, Union[ResponsesAPIResponse, AsyncIterator[Any]]],
    ]:
        if not _is_async:
            raise ValueError(
                "The Responses -> Anthropic /v1/messages bridge is async only; "
                "call litellm.aresponses()."
            )
        return LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
            model=model,
            input=input,
            responses_api_request=responses_api_request,
            custom_llm_provider=custom_llm_provider,
            stream=stream,
            **kwargs,
        )

    @staticmethod
    async def async_response_api_handler(
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, AsyncIterator[Any]]:
        from collections.abc import AsyncIterator as AsyncIteratorABC

        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        translated = translate_responses_request_to_anthropic_messages_request(
            model=model,
            input=input,
            responses_api_request=responses_api_request,
        )
        request: Dict[str, Any] = dict(translated.request)
        request.pop("model", None)
        request.pop("stream", None)

        created_at = int(time.time())
        # ``is_async`` makes the handler return a coroutine; going through it
        # rather than litellm.anthropic.messages.acreate keeps force_reasoning_effort
        # and reasoning_auto_summary while avoiding a second @client logging wrapper.
        response = await cast(
            Any,
            anthropic_messages_handler(
                model=model,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                is_async=True,
                **request,
                **kwargs,
            ),
        )

        if isinstance(response, AsyncIteratorABC):
            iterator = AnthropicMessagesToResponsesStreamIterator(
                anthropic_stream=cast(AsyncIterator[bytes], response),
                model=model,
                tool_context=translated.tool_context,
                responses_api_request=responses_api_request,
                created_at=created_at,
                responses_tools=translated.responses_tools,
            )
            logging_obj = kwargs.get("litellm_logging_obj")
            if logging_obj is None:
                return iterator

            from litellm.responses.streaming_iterator import (
                ResponsesAPIEventStreamIterator,
            )

            return ResponsesAPIEventStreamIterator(
                event_iterator=iterator,
                model=model,
                logging_obj=logging_obj,
                litellm_metadata=kwargs.get("litellm_metadata"),
                custom_llm_provider=custom_llm_provider,
                request_data=kwargs,
                call_type="aresponses",
            )

        return translate_anthropic_messages_response_to_responses_api_response(
            response=cast(AnthropicMessagesResponse, response),
            tool_context=translated.tool_context,
            request=responses_api_request,
            created_at=created_at,
            responses_tools=translated.responses_tools,
        )
