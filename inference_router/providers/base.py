from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator
from inference_router.models import RouterRequest, RouterResponse


class BaseProvider(ABC):
    """
    Abstract base class for all providers.

    To support any model from any API, subclass this and implement:
        - name property  — unique identifier for this provider+model
        - complete()     — sync completion
        - acomplete()    — async completion

    Optionally override:
        - stream()       — sync streaming (defaults to complete())
        - astream()      — async streaming (defaults to acomplete())
        - health_check() — liveness check (defaults to True)

    Example minimal custom provider:

        class MyProvider(BaseProvider):

            def __init__(self, api_key: str, model: str):
                self.api_key = api_key
                self.model = model

            @property
            def name(self) -> str:
                return f"myprovider/{self.model}"

            def complete(self, request: RouterRequest) -> RouterResponse:
                # call your API here
                ...

            async def acomplete(self, request: RouterRequest) -> RouterResponse:
                # async version
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this provider and model.
        Shows up in logs, stats, and RouterResponse.model_used.

        Convention: "provider/model-name"
        Examples:
            "bedrock/anthropic.claude-haiku-4-5"
            "openai/gpt-4o"
            "ollama/llama3"
            "deepinfra/meta-llama-3-8b"
        """
        ...

    @abstractmethod
    def complete(self, request: RouterRequest) -> RouterResponse:
        """
        Synchronous completion. Calls the model and returns
        a normalized RouterResponse.

        The router measures latency around this call automatically,
        so you don't need to measure it yourself.

        Args:
            request: RouterRequest with prompt, messages,
                     max_tokens, temperature, system_prompt

        Returns:
            RouterResponse — normalized shape, same for every provider
        """
        ...

    @abstractmethod
    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        """
        Async completion. Same as complete() but non-blocking.
        Use this in FastAPI endpoints and async applications.

        Args:
            request: RouterRequest

        Returns:
            RouterResponse
        """
        ...

    def stream(self, request: RouterRequest) -> Iterator[str]:
        """
        Synchronous streaming. Yields text chunks as they arrive.

        Default behavior: falls back to complete() and yields
        the full response as a single chunk. Override this in
        your provider if the API supports real streaming.

        Args:
            request: RouterRequest

        Yields:
            str — chunks of the response text
        """
        response = self.complete(request)
        yield response.text

    async def astream(self, request: RouterRequest) -> AsyncIterator[str]:
        """
        Async streaming. Yields text chunks as they arrive.

        Default behavior: falls back to acomplete() and yields
        the full response as a single chunk. Override this in
        your provider if the API supports real streaming.

        Args:
            request: RouterRequest

        Yields:
            str — chunks of the response text
        """
        response = await self.acomplete(request)
        yield response.text

    def health_check(self) -> bool:
        """
        Check if this provider is reachable and healthy.

        Used by the latency strategy and fallback logic to avoid
        routing to a provider that is down or rate-limited.

        Default: always returns True.
        Override for real liveness checks.

        Returns:
            bool — True if provider is healthy, False otherwise
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"