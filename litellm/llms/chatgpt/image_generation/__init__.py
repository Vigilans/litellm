from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import ChatGPTImageGenerationConfig

__all__ = [
    "ChatGPTImageGenerationConfig",
]


def get_chatgpt_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return ChatGPTImageGenerationConfig()
