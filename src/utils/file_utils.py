# file_utils.py (优化后)
import os
import re
from typing import Union
from pathlib import Path
from pydantic import BaseModel
from ..schemas import order_fields
from .model_utils import (
    load_audio_model,
    load_vision_model,
    load_text_model,
    get_available_models  # 新增获取模型状态接口
)

from traceback import print_exc


# 支持的文件类型
AUDIO_EXTS = {'.wav', '.mp3', '.amr'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
TEXT_EXTS = {'.txt'}

def determine_input_type(input_data: Union[str, Path]) -> str:
    """自动判断输入数据类型（优化文件存在性检查）"""
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix in AUDIO_EXTS:
                return 'audio'
            if suffix in IMAGE_EXTS:
                return 'image'
            if suffix in TEXT_EXTS:
                return 'text'
    return 'text'

def unified_process(input_data: Union[str, Path, bytes]):
    """
    统一处理入口函数（增加模型预热逻辑）
    """
    try:
        # 预加载所有模型（按需加载）
        _ = get_available_models()  # 触发模型预加载
        
        # 类型判断逻辑优化
        input_type = 'text'
        file_path = None
        
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.exists():
                file_path = path
                input_type = determine_input_type(path)

        # 分派处理逻辑
        if input_type == 'audio':
            return process_audio(file_path)
        elif input_type == 'image':
            return process_image(file_path)
        else:
            text_content = file_path.read_text(encoding='utf-8') if file_path else str(input_data)
            return process_text(text_content)

    except Exception as e:
        print_exc()
        return handle_processing_error(input_data, str(e))

def process_audio(audio_path: Path):
    """处理音频文件（使用持久化模型）"""
    from ..processing.audio_processor import AudioProcessor
    
    try:
        # 直接使用已加载的模型
        processor = load_audio_model()
        result = processor.process(str(audio_path))
        
        return result
    except Exception as e:
        print_exc()
        return handle_processing_error(audio_path, str(e))

def process_image(image_path: Path):
    """处理图片文件（使用持久化模型）"""
    from ..processing.image_processor import LogisticsExtractor
    
    try:
        # 直接使用已加载的模型
        processor = load_vision_model()
        return processor.extract_from_image(str(image_path))
    except Exception as e:
        print_exc()
        return handle_processing_error(image_path, str(e))

def process_text(text: str):
    """处理文本内容（使用持久化模型）"""
    from ..processing.text_processor import TextProcessor
    
    try:
        # 直接使用已加载的模型
        processor = load_text_model()
        result = processor.process(text)
        
        return result
    except Exception as e:
        print_exc()
        return handle_processing_error(text, str(e))

# 其余函数保持不变...
