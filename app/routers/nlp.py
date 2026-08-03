from fastapi import APIRouter, Request
from app.schemas import SentimentRequest, SentimentResponse, Sentiment
from app.services import nlp_service

router = APIRouter(tags=["Text Analytics"])

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def api_analyze_sentiment(request: Request, payload: SentimentRequest):
    """Endpoint to classify customer feedback into positive, neutral, or negative sentiment."""
    sentiment_label, conf_score = await nlp_service.predict_sentiment(payload.text)
    
    try:
        mapped_sentiment = Sentiment(sentiment_label)
    except ValueError:
        mapped_sentiment = Sentiment.NEUTRAL
        
    result_payload = SentimentResponse(
        text=payload.text,
        sentiment=mapped_sentiment,
        confidence=conf_score
    )
    
    request.app.state.sentiment_log.append(result_payload.model_dump())
    
    return result_payload
