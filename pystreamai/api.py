"""PyStreamAI HTTP API Server (FastAPI)"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from .serving import InferenceServer

logger = logging.getLogger(__name__)


# Request/Response models
class PredictRequest(BaseModel):
    """Prediction request"""
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    """Prediction response"""
    request_id: str
    model_id: str
    output: Any
    latency_ms: float
    batch_size: int
    cost_usd: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    uptime_seconds: float
    requests_processed: int


class StatsResponse(BaseModel):
    """Server statistics"""
    requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_cost_usd: float
    uptime_seconds: float


class APIServer:
    """HTTP API wrapper around InferenceServer.

    `model` is required and loaded into the InferenceServer immediately
    (synchronously, via asyncio.run -- safe here since this constructor
    runs before uvicorn starts an event loop). Previously this class
    accepted only a model_id string with no way to provide an actual
    model, so /predict always hit InferenceServer's old fabricated
    response path; now that InferenceServer requires a real loaded model,
    that gap had to be closed here too.
    """

    def __init__(self, model_id: str, model: Any, gpu_type: str = "A100", num_gpus: int = 1, port: int = 8000):
        self.model_id = model_id
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.port = port

        # Initialize FastAPI app
        self.app = FastAPI(
            title="PyStreamAI",
            description="Simple ML Inference Server",
            version="0.1.0",
        )

        # Initialize inference server and load the real model.
        self.server = InferenceServer(gpu_type, num_gpus)
        asyncio.run(self.server.load_model(model_id, model))

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check"""
            health_info = self.server.health_check()
            return HealthResponse(**health_info)

        @self.app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            """Run inference"""
            try:
                response = await self.server.predict(self.model_id, request.data)
                predict_response = PredictResponse(**response)
                # FastAPI serializes the return value *after* this handler
                # returns, outside this try/except -- a model whose output
                # isn't JSON-serializable (a custom object with no
                # pydantic-compatible representation, a raw file handle,
                # etc.) would otherwise surface as an unhandled
                # PydanticSerializationError / 500 with no clean `detail`,
                # unlike every other inference failure here. Forcing the
                # same serialization now makes that failure mode go through
                # this handler's own error path instead.
                predict_response.model_dump_json()
                return predict_response
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats", response_model=StatsResponse)
        async def stats():
            """Get server statistics"""
            stats_data = self.server.get_stats()
            return StatsResponse(**stats_data)

        @self.app.get("/models/{model_id}/info")
        async def model_info(model_id: str):
            """Get model information"""
            if model_id != self.model_id:
                raise HTTPException(status_code=404, detail="Model not found")

            return {
                "model_id": model_id,
                "gpu_type": self.gpu_type,
                "num_gpus": self.num_gpus,
                "status": "ready",
            }

        @self.app.post("/shutdown")
        async def shutdown(background_tasks: BackgroundTasks):
            """Graceful shutdown"""
            background_tasks.add_task(self._shutdown)
            return {"status": "shutting down"}

    async def _shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Shutting down server...")
        # Cleanup code here

    def run(self):
        """Start the API server"""
        logger.info(f"Starting PyStreamAI API server on port {self.port}")
        logger.info(f"Model: {self.model_id}")
        logger.info(f"GPU: {self.gpu_type} x{self.num_gpus}")
        logger.info(f"Health check: http://localhost:{self.port}/health")
        logger.info(f"Predict: POST http://localhost:{self.port}/predict")
        logger.info(f"Stats: http://localhost:{self.port}/stats")

        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")


def create_api_server(
    model_id: str,
    model: Any,
    gpu_type: str = "A100",
    num_gpus: int = 1,
    port: int = 8000,
) -> APIServer:
    """Create and return an API server serving real inference for `model`.

    `model` must be a real .onnx path/ONNXModelLoader, have a callable
    `.predict()`, or be callable itself -- see
    pystreamai.platform.Endpoint's docstring. Raises TypeError
    immediately (via InferenceServer.load_model) otherwise.
    """
    return APIServer(model_id, model, gpu_type, num_gpus, port)
