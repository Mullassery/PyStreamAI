"""PyStreamAI Decorators - @train, @serve, @pipeline"""

import functools
from typing import Callable, Optional, Any
from .platform import get_platform, TrainingJob, Endpoint


def train(
    gpu: Optional[str] = None,
    time_limit: Optional[str] = None,
    **train_kwargs
) -> Callable:
    """
    Decorator to mark a function as a training job.

    @train(gpu="A100", time_limit="1h")
    def train_model(data):
        model = train_bert(data)
        return model
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> TrainingJob:
            platform = get_platform()
            model_id = train_kwargs.get("model_id", fn.__name__)

            # Execute the training function
            model = fn(*args, **kwargs)

            # Submit to platform
            job = platform.train(
                code=fn.__module__,
                dataset="inline",
                gpu=gpu,
                time_limit=time_limit,
                model_id=model_id,
            )

            # Store model artifact
            print(f"   Model trained: {model_id}")

            return job

        return wrapper
    return decorator


def serve(
    replicas: int = 1,
    gpu: Optional[str] = None,
    max_batch_size: Optional[int] = None,
    **serve_kwargs
) -> Callable:
    """
    Decorator to mark a function as a serving endpoint.

    @serve(replicas=3, gpu="L4")
    def predict(x):
        return model.predict(x)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Endpoint:
            platform = get_platform()
            model_id = serve_kwargs.get("model_id", fn.__name__)

            # Create endpoint
            endpoint = platform.serve(
                model=fn,
                replicas=replicas,
                gpu=gpu,
                max_batch_size=max_batch_size,
                model_id=model_id,
            )

            # Wrap the original function to use the endpoint
            original_fn = fn

            def predict_wrapper(data: Any) -> Any:
                result = original_fn(data)
                return result

            endpoint.predict = predict_wrapper

            return endpoint

        return wrapper
    return decorator


def pipeline(
    name: Optional[str] = None,
    **pipeline_kwargs
) -> Callable:
    """
    Decorator to mark a function as a pipeline.

    @pipeline(name="training-pipeline")
    def workflow():
        data = preprocess()
        model = train(data)
        serve(model)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            pipeline_name = name or fn.__name__
            print(f"🔗 Running pipeline: {pipeline_name}")
            result = fn(*args, **kwargs)
            print(f"✅ Pipeline {pipeline_name} completed")
            return result

        return wrapper
    return decorator
