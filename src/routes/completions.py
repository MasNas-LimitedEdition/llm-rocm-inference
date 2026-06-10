"""
/v1/completions — OpenAI-compatible text completion endpoint
"""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


@router.post("/completions")
async def completions(request: Request, body: CompletionRequest):
    engine = request.app.state.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    final_output = None
    async for output in engine.generate(
        prompt=body.prompt,
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
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "finish_reason": final_output.outputs[0].finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
