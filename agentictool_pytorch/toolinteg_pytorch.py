from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import time
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = "1.2"
MODEL_PATH = "fault_model_v1.2.pth"

class FaultClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.network(x)

# Load PyTorch model weights
model = FaultClassifier()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_loaded = False

class PredictRequest(BaseModel):
    temperature: float
    vibration: float
    pressure: float

class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
    latency_microseconds: float
    status: int

class ErrorResponse(BaseModel):
    error: str
    status: int

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation failed: request body must contain numeric temperature, vibration, and pressure", "status": 400}
    )

# TODO: Implement endpoints and validation
# Create a GET /health endpoint to check service status
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded, "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(request: PredictRequest):
    # Convert request values to a tensor, run torch.no_grad() inference,
    # apply torch.softmax(), and return prediction + confidence.
    try:
        input_tensor = torch.tensor([[request.temperature, request.vibration, request.pressure]], dtype=torch.float32)
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        latency_microseconds = (time.time() - start_time) * 1e6
        return PredictResponse(
            prediction=int(predicted_class.item()),
            confidence=float(confidence.item()),
            model_version=MODEL_VERSION,
            latency_microseconds=latency_microseconds,
            status=200
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def main():
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

main()