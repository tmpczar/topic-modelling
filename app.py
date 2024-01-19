from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.infer import predict
import gradio as gr
from typing import Tuple

app = FastAPI()


class Request(BaseModel):
    text: str


class Response(BaseModel):
    topic: str
    confidence: float


@app.post("/predict_topic", response_model=Response)
async def predict_topic(request: Request) -> Response:
    """Endpoint for predictions using the fine tuned BERTopic model"""

    try:
        result = predict(request.text)  # make sure to use request.text, not Request.text
        return Response(
            topic=result["predicted_topic"], confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def gradio_predict(text: str) -> Tuple[str, float]:
    result = predict(text)
    return result["predicted_topic"], result["confidence"]


interface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Enter Your Text",
        placeholder="Enter an open-ended question to find its topic",
    ),
    outputs=[gr.Textbox(label="Predicted Topic"), gr.Number(label="Confidence Score")],
    allow_flagging="never",
)

app = gr.mount_gradio_app(app, interface, path='/')