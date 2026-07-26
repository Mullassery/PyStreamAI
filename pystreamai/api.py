"""PyStreamAI HTTP API Server (FastAPI)"""

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
    """HTTP API wrapper around InferenceServer"""

    def __init__(self, model_id: str, gpu_type: str = "A100", num_gpus: int = 1, port: int = 8000):
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

        # Initialize inference server
        self.server = InferenceServer(gpu_type, num_gpus)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check"""
            health_info = self.server.health_check()
            return HealthResponse(**health_info)

        @self.app.post(f"/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            """Run inference"""
            try:
                response = await self.server.predict(self.model_id, request.data)
                return PredictResponse(**response)
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
    gpu_type: str = "A100",
    num_gpus: int = 1,
    port: int = 8000,
) -> APIServer:
    """Create and return API server"""
    return APIServer(model_id, gpu_type, num_gpus, port)
