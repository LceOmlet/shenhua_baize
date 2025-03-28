# model_utils.py
import torch
import os
from functools import lru_cache
from typing import Tuple, Any, Dict

# Model cache
MODEL_CACHE = {
    "audio": None,
    "vision": None,
    "text": None
}

def init_all_models():
    """Unified initialization of all models with model sharing and 4-bit quantization"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import whisper
    from ..utils.config_utils import config
    
    # Set CUDA memory allocator
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Get configurations
    SPEECH_CONFIG = config.get("speech_model_config", {})
    WHISPER_CONFIG = config.get("whisper_config", {})
    VISION_CONFIG = config.get("vision_model_config", {})
    TEXT_CONFIG = config.get("text_model_config", {})
    
    # Get model paths
    speech_model_path = SPEECH_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")
    text_model_path = TEXT_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")
    vision_model_path = VISION_CONFIG.get("model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Create cache for shared models
    model_cache = {}
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Check if all models are using the same VL model
    using_same_vl_model = speech_model_path == vision_model_path and text_model_path == vision_model_path
    
    # If using the same VL model, initialize vision model first
    if using_same_vl_model:
        # ---------- Initialize Vision Model First ----------
        # Setup vision device
        vision_cuda_devices = VISION_CONFIG.get("cuda_devices", "0")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            selected_gpu = min(int(vision_cuda_devices), num_gpus-1) if vision_cuda_devices.isdigit() else 0
            vision_device = f"cuda:{selected_gpu}"
        else:
            vision_device = "cpu"
        
        # Initialize vision model
        vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vision_model_path,
            quantization_config=quantization_config,
            device_map={"": vision_device} if vision_device != "auto" else "auto",
            trust_remote_code=True
        ).eval()
        
        # Initialize vision processor
        vision_processor = AutoProcessor.from_pretrained(
            vision_model_path,
            trust_remote_code=True
        )
        
        # Share VL model with text and speech
        speech_model = vision_model
        speech_tokenizer = vision_processor
        text_model = vision_model
        text_tokenizer = vision_processor
        
        # ---------- Initialize Whisper Model ----------
        whisper_device = f"cuda:{SPEECH_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(WHISPER_CONFIG.get("model_name", "base"))
        if torch.cuda.is_available():
            whisper_model = whisper_model.to(whisper_device)
    else:
        # ---------- Initialize Speech/Text Models ----------
        def init_speech_text_model(config, model_type):
            cuda_devices = config.get("cuda_devices", "0")
            model_path = config.get("model_path", "Qwen/Qwen2.5-7B-Instruct")
            
            # Share model if possible
            if model_type == "text" and model_cache.get("shared_model"):
                return model_cache["shared_model"], model_cache["shared_tokenizer"]
            
            # Device configuration
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                selected_gpu = min(int(cuda_devices), num_gpus-1) if cuda_devices.isdigit() else 0
                model_device = f"cuda:{selected_gpu}"
            else:
                model_device = "cpu"
                
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Initialize model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map={"": model_device},
                trust_remote_code=True
            )
            
            # Cache shared model
            if model_type == "speech" and speech_model_path == text_model_path:
                model_cache["shared_model"] = model
                model_cache["shared_tokenizer"] = tokenizer
                
            return model, tokenizer
        
        # Initialize speech model
        speech_model, speech_tokenizer = init_speech_text_model(SPEECH_CONFIG, "speech")
        
        # Initialize text model (shared if possible)
        if speech_model_path == text_model_path:
            text_model, text_tokenizer = model_cache["shared_model"], model_cache["shared_tokenizer"]
        else:
            text_model, text_tokenizer = init_speech_text_model(TEXT_CONFIG, "text")
        
        # ---------- Initialize Whisper Model ----------
        whisper_device = f"cuda:{SPEECH_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(WHISPER_CONFIG.get("model_name", "base"))
        if torch.cuda.is_available():
            whisper_model = whisper_model.to(whisper_device)
        
        # ---------- Initialize Vision Model ----------
        # Get vision model path and device configuration from settings
        vision_cuda_devices = VISION_CONFIG.get("cuda_devices", "0")
        
        # Setup vision device
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            selected_gpu = min(int(vision_cuda_devices), num_gpus-1) if vision_cuda_devices.isdigit() else 0
            vision_device = f"cuda:{selected_gpu}"
        else:
            vision_device = "cpu"
        
        # Initialize vision model
        vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vision_model_path,
            quantization_config=quantization_config,
            device_map={"": vision_device} if vision_device != "auto" else "auto",
            trust_remote_code=True
        ).eval()
        
        # Initialize vision processor
        vision_processor = AutoProcessor.from_pretrained(
            vision_model_path,
            trust_remote_code=True
        )
    
    return {
        "speech": (speech_model, speech_tokenizer, whisper_model),
        "vision": (vision_model, vision_processor),
        "text": (text_model, text_tokenizer)
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

