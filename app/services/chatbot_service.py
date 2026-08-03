import os
import json
import joblib
from typing import Tuple

_INTENT_MODEL_PATH = "app/models/chatbot_model.pkl"
_INTENT_VECTORIZER_PATH = "app/models/chatbot_vectorizer.pkl"
_INTENTS_JSON_PATH = "data/intents.json"

async def get_chatbot_reply(user_message: str) -> Tuple[str, str, float]:
    message_lower = user_message.lower()
    
    if "return" in message_lower and "policy" in message_lower:
        return "You can return any unworn items within 30 days of purchase.", "return_policy", 1.0
    if "hours" in message_lower or "open" in message_lower:
        return "Our stores are open Monday through Saturday from 9 AM to 9 PM.", "store_hours", 1.0
    if "order" in message_lower and "status" in message_lower:
        return "Please provide your order number, and I will check the status for you.", "order_status", 1.0
        
    if os.path.exists(_INTENT_MODEL_PATH) and os.path.exists(_INTENT_VECTORIZER_PATH):
        try:
            vectorizer = joblib.load(_INTENT_VECTORIZER_PATH)
            clf = joblib.load(_INTENT_MODEL_PATH)
            
            vec_text = vectorizer.transform([message_lower])
            intent = clf.predict(vec_text)[0]
            conf = max(clf.predict_proba(vec_text)[0])
            
            return f"Simulated ML response for intent: {intent}", str(intent), float(conf)
        except Exception:
            pass
            
    return "I am a virtual assistant. Could you please rephrase your question or contact support?", "fallback", 0.5
