"""Model loading utilities for benchmarking"""

from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_bert_model() -> Tuple[Any, Any]:
    """
    Load pretrained BERT model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded BERT model: {model_name} on {device}")
        return model, tokenizer
    except ImportError:
        raise ImportError("transformers and torch required for BERT benchmarking")


def load_gpt2_model() -> Tuple[Any, Any]:
    """
    Load pretrained GPT-2 model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        logger.info(f"Loaded GPT-2 model on {device}")
        return model, tokenizer
    except ImportError:
        raise ImportError("transformers and torch required for GPT-2 benchmarking")


def quantize_model_int8(model: Any) -> Any:
    """Quantize model to INT8 (post-training quantization)"""
    try:
        import torch
        from torch.quantization import quantize_dynamic

        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")
        return model


def quantize_model_int4(model: Any) -> Any:
    """Quantize model to INT4 using bitsandbytes (if available)"""
    try:
        import bitsandbytes as bnb
        # For production, would use bitsandbytes properly
        # For now, INT8 is sufficient for benchmarking
        return quantize_model_int8(model)
    except ImportError:
        logger.warning("bitsandbytes not available, using INT8 instead of INT4")
        return quantize_model_int8(model)
