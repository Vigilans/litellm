from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ImageResponse

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ChatGPTImageGenerationConfig(GPTImageGenerationConfig):
    def __init__(self, authenticator: Optional[Authenticator] = None) -> None:
        super().__init__()
        self.authenticator = authenticator or Authenticator()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        return f"{api_base.rstrip('/')}/images/generations"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        default_headers = get_chatgpt_default_headers(
            access_token,
            self.authenticator.get_account_id(),
            ensure_chatgpt_session_id(litellm_params),
        )
        default_headers["accept"] = "application/json"
        return {**default_headers, **headers}

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"model": model, "prompt": prompt, **optional_params}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        image_response = super().transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # ChatGPT resolves these server-side and ignores unsupported requested
        # values, so the echoed request params the parent sets are misleading.
        response = raw_response.json()
        for field in ("size", "quality", "output_format"):
            if response.get(field) is not None:
                setattr(image_response, field, response[field])

        return image_response
