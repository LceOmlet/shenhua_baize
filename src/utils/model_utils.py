from ..utils.config_utils import load_config
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

config = load_config()


def init_all_models():
    """统一初始化所有模态模型"""
    config = load_config()
    
    # 获取各模块配置
    SPEECH_CONFIG = config.get("speech_model_config", {})
    WHISPER_CONFIG = config.get("whisper_config", {})
    VISION_CONFIG = config.get("vision_model_config", {})
    TEXT_CONFIG = config.get("text_model_config", {})

    #---------- 共享模型检查 ----------
    speech_model_path = SPEECH_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")
    text_model_path = TEXT_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")
    model_cache = {}

    #---------- 语音/文本模型初始化 ----------
    def init_speech_text_model(config, model_type):
        cuda_devices = config.get("cuda_devices", "0")
        
        # 共享模型逻辑
        if model_type == "text" and model_cache.get("shared_model"):
            return model_cache["shared_model"], model_cache["shared_tokenizer"]

        device = f"cuda:{TEXT_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], 
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map={"": device},
            trust_remote_code=True
        )

        # 缓存共享模型
        if model_type == "speech" and speech_model_path == text_model_path:
            model_cache["shared_model"] = model
            model_cache["shared_tokenizer"] = tokenizer

        return model, tokenizer

    # 初始化语音模型
    if "speech_model" not in model_cache:
        speech_model, speech_tokenizer = init_speech_text_model(SPEECH_CONFIG, "speech")
    
    # 初始化文本模型
    if speech_model_path == text_model_path:
        text_model, text_tokenizer = model_cache["shared_model"], model_cache["shared_tokenizer"]
    else:
        text_model, text_tokenizer = init_speech_text_model(TEXT_CONFIG, "text")

    #---------- Whisper模型初始化 ----------
    whisper_device = f"cuda:{SPEECH_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model(WHISPER_CONFIG.get("model_name", "base"))
    if "cuda" in whisper_device:
        whisper_model = whisper_model.to(whisper_device)

    #---------- 视觉模型初始化 ----------
    vision_device = f"cuda:{VISION_CONFIG.get('cuda_devices', '0')}" if torch.cuda.is_available() else "cpu"
    vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VISION_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct"),
        torch_dtype=torch.float16,
        device_map={"": vision_device},
        trust_remote_code=True
    ).eval()

    vision_processor = AutoProcessor.from_pretrained(
        VISION_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct"),
        trust_remote_code=True
    )

    return {
        "speech": (speech_model, speech_tokenizer, whisper_model),
        "vision": (vision_model, vision_processor),
        "text": (text_model, text_tokenizer)
    }

models = init_all_models()
SPEECH_MODEL = models['speech']
VISION_MODEL = models['vision']
TEXT_MODEL = models['text']
