import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.utils.config_utils import config

# 全局缓存
_cached_models = {}

def get_text_model():
    """持久化并获取文本处理模型"""
    if "text_model" not in _cached_models:
        MODEL_PATH = config.get("text_model_config", {}).get("model_path", "Qwen/Qwen2.5-7B-Instruct")
        _cached_models["text_model"] = {
            "model": AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto"),
            "tokenizer": AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        }
    return _cached_models["text_model"]

def get_audio_model():
    """持久化并获取 Whisper 语音模型"""
    if "whisper_model" not in _cached_models:
        WHISPER_MODEL = config.get("whisper_config", {}).get("model_name", "base")
        _cached_models["whisper_model"] = whisper.load_model(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return _cached_models["whisper_model"]

def get_vision_model():
    """持久化并获取视觉处理模型"""
    if "vision_model" not in _cached_models:
        VISION_MODEL_PATH = config.get("vision_model_config", {}).get("model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
        _cached_models["vision_model"] = {
            "model": Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VISION_MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            ).eval(),
            "processor": AutoProcessor.from_pretrained(VISION_MODEL_PATH, trust_remote_code=True)
        }
    return _cached_models["vision_model"]
