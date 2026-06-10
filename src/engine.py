"""
vLLM Engine Wrapper — AMD ROCm backend
"""

import asyncio
import logging
from typing import AsyncGenerator, List, Optional

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput

from utils.logger import setup_logger

logger = setup_logger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        self.model = model
        self.engine_args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            # ROCm-specific optimizations
            worker_use_ray=tensor_parallel_size > 1,
            engine_use_ray=False,
        )
        self._engine: Optional[AsyncLLMEngine] = None
        self._request_counter = 0

    async def initialize(self):
        logger.info(f"Loading model: {self.model}")
        self._engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        logger.info("Engine initialized successfully")

    async def shutdown(self):
        if self._engine:
            await self._engine.shutdown_background_loop()

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate text from a prompt."""
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
        )

        request_id = self._next_request_id()
        results_generator = self._engine.generate(prompt, sampling_params, request_id)

        async for request_output in results_generator:
            yield request_output
            if not stream and request_output.finished:
                break

    async def generate_chat(
        self,
        messages: List[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate from chat messages (applies chat template)."""
        prompt = self._apply_chat_template(messages)
        async for output in self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
        ):
            yield output

    def _apply_chat_template(self, messages: List[dict]) -> str:
        """Simple chat template — models like Mistral/LLaMA use [INST] format."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                formatted += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content} </s>"
        return formatted
