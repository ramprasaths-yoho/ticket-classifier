from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.model import load_predictor


class Ticket(BaseModel):
    text: str = Field(..., max_length=5000, examples=["I was charged twice"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class Prediction(BaseModel):
    category: str
    priority: str | None = None
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()  # loaded ONCE, not per request
    yield


app = FastAPI(title="Support Ticket Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    return {"status": "ok" if predictor else "unavailable", "model": type(predictor).__name__}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket, request: Request):
    return request.app.state.predictor.predict([ticket.text])[0]


# Bonus: batch endpoint. Body is a JSON list of tickets.
@app.post("/predict-batch", response_model=list[Prediction])
def predict_batch(tickets: list[Ticket], request: Request):
    if not tickets:
        raise HTTPException(status_code=422, detail="tickets list must not be empty")
    if len(tickets) > 100:
        raise HTTPException(status_code=422, detail="max 100 tickets per request")
    return request.app.state.predictor.predict([t.text for t in tickets])
