"""
/v1/chat/completions — OpenAI-compatible chat endpoint
"""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    engine = request.app.state.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        async def stream_generator():
            async for output in engine.generate_chat(
                messages=messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                stop=body.stop,
                stream=True,
            ):
                if output.outputs:
                    delta = output.outputs[0].text
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": body.model,
                        "choices": [{"delta": {"content": delta}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming
    final_output = None
    async for output in engine.generate_chat(
        messages=messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop=body.stop,
        stream=False,
    ):
        final_output = output

    if final_output is None or not final_output.outputs:
        raise HTTPException(status_code=500, detail="No output generated")

    generated_text = final_output.outputs[0].text
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = len(final_output.outputs[0].token_ids)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": final_output.outputs[0].finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
