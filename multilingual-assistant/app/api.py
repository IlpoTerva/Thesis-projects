from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import ChatResponse, ChatRequest
import json
from app.chat_lang import generate_chat_response_ui, stream_chat_response

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        
        return generate_chat_response_ui(message=payload.message, 
                                        session_id=payload.session_id,
                                        choide_id=payload.choice_id
                                        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    def events():
        try:
            for event in stream_chat_response(message=payload.message, session_id=payload.session_id, choide_id=payload.choice_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")