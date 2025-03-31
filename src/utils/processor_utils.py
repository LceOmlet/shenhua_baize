# model_utils.py
import torch
import os
from functools import lru_cache
from typing import Tuple, Any, Dict
from vllm.engine.arg_utils import EngineArgs
from vllm import LLM
from transformers import AutoProcessor
import whisper
from ..utils.config_utils import config
import torch
torch.cuda.empty_cache()


# Model cache
MODEL_CACHE = {
    "audio": None,
    "vision": None,
    "text": None
}

def init_all_models():
    """使用vLLM初始化所有模型"""
    from transformers import AutoTokenizer
    from ..utils.config_utils import config

    import torch
    torch.cuda.empty_cache()

    # Set CUDA memory allocator
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Get configurations
    SPEECH_CONFIG = config.get("speech_model_config", {})
    WHISPER_CONFIG = config.get("whisper_config", {})
    VISION_CONFIG = config.get("vision_model_config", {})
    TEXT_CONFIG = config.get("text_model_config", {})
    
    # Get model paths - 使用更小的模型版本
    speech_model_path = SPEECH_CONFIG.get("model_path", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    text_model_path = TEXT_CONFIG.get("model_path", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    vision_model_path = VISION_CONFIG.get("model_path", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    
    # 初始化vLLM模型
    engine = LLM(
        model=vision_model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        quantization="awq",
        dtype="float16",
        tensor_parallel_size=1,
        enforce_eager=True,
        disable_log_stats=True,
    )
    
    # 初始化处理器
    processor = AutoProcessor.from_pretrained(
        vision_model_path,
        trust_remote_code=True
    )
    
    # 初始化Whisper模型
    whisper_device = f"cuda:{SPEECH_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(WHISPER_CONFIG.get("model_name", "base"))
    if torch.cuda.is_available():
        whisper_model = whisper_model.to(whisper_device)
    
    return {
        "speech": (engine, processor, whisper_model),
        "vision": (engine, processor),
        "text": (engine, processor)
    }

# Initialize all models once at module load time
MODELS = init_all_models()

def init_vision_model():
    """Get vision model and processor"""
    return MODELS["vision"]

def init_audio_model():
    """Get audio models"""
    return MODELS["speech"]

def init_text_model():
    """Get text model"""
    return MODELS["text"]

@lru_cache(maxsize=1)
def load_audio_model():
    """Load and cache audio model"""
    if MODEL_CACHE["audio"] is None:
        from ..processing.audio_processor import AudioProcessor
        MODEL_CACHE["audio"] = AudioProcessor()
    return MODEL_CACHE["audio"]

@lru_cache(maxsize=1)
def load_vision_model():
    """Load and cache vision model"""
    if MODEL_CACHE["vision"] is None:
        from ..processing.image_processor import LogisticsExtractor
        MODEL_CACHE["vision"] = LogisticsExtractor()
    return MODEL_CACHE["vision"]

@lru_cache(maxsize=1)
def load_text_model():
    """Load and cache text model"""
    if MODEL_CACHE["text"] is None:
        from ..processing.text_processor import TextProcessor
        MODEL_CACHE["text"] = TextProcessor()
    return MODEL_CACHE["text"]

def get_available_models() -> dict:
    """Get information about currently loaded models"""
    return {
        "audio": "loaded" if MODEL_CACHE["audio"] else "unloaded",
        "vision": "loaded" if MODEL_CACHE["vision"] else "unloaded",
        "text": "loaded" if MODEL_CACHE["text"] else "unloaded"
    }

