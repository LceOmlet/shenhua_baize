# model_utils.py
import torch
from functools import lru_cache

# 模型缓存
MODEL_CACHE = {
    "audio": None,
    "vision": None,
    "text": None
}

@lru_cache(maxsize=1)
def load_audio_model():
    """加载并缓存语音模型"""
    if MODEL_CACHE["audio"] is None:
        from ..processing.audio_processor import AudioProcessor
        MODEL_CACHE["audio"] = AudioProcessor()
    return MODEL_CACHE["audio"]

@lru_cache(maxsize=1)
def load_vision_model():
    """加载并缓存视觉模型"""
    if MODEL_CACHE["vision"] is None:
        from ..processing.image_processor import LogisticsExtractor
        MODEL_CACHE["vision"] = LogisticsExtractor()
    return MODEL_CACHE["vision"]

@lru_cache(maxsize=1)
def load_text_model():
    """加载并缓存文本模型"""
    if MODEL_CACHE["text"] is None:
        from ..processing.text_processor import TextProcessor
        MODEL_CACHE["text"] = TextProcessor()
    return MODEL_CACHE["text"]

def get_available_models() -> dict:
    """获取当前加载的模型信息"""
    return {
        "audio": "loaded" if MODEL_CACHE["audio"] else "unloaded",
        "vision": "loaded" if MODEL_CACHE["vision"] else "unloaded",
        "text": "loaded" if MODEL_CACHE["text"] else "unloaded"
    }
