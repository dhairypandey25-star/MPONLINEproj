from fastapi import APIRouter, Request
from app.schemas import ChatbotRequest, ChatbotResponse
from app.services import chatbot_service

router = APIRouter(tags=["Customer Support"])

@router.post("/chatbot", response_model=ChatbotResponse)
async def api_chat_with_bot(request: Request, payload: ChatbotRequest):
    reply_text, detected_intent, conf_score = await chatbot_service.get_chatbot_reply(payload.message)
    
    result_payload = ChatbotResponse(
        reply=reply_text,
        intent=detected_intent,
        confidence=conf_score
    )
    
    log_entry = {
        "user_message": payload.message,
        "bot_reply": reply_text,
        "intent": detected_intent
    }
    request.app.state.chatbot_log.append(log_entry)
    
    return result_payload
