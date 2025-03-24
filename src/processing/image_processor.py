# logistics_vision.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from pydantic import BaseModel
import yaml
import re
import json
import os
import torch
from typing import Optional
from datetime import datetime
from ..schemas import order_fields
from ..utils.config_utils import config
from ..utils.config_utils import load_config
from ..schemas import order_fields, ExtractionResult
from ..utils.model_utils import VISION_MODEL
from ..utils.prompt_utils import build_prompt
# ---------- 根据项目结构调整的模块导入 ----------
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pydantic import BaseModel

# ---------- 模型初始化 ----------
def init_models():
    """初始化视觉模型和处理器"""
    # 设置 CUDA 内存分配器
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 配置4-bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 使用4-bit量化加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.get('vision_model_path'),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        config.get('vision_model_path'),
        trust_remote_code=True
    )

    return model, processor

# ---------- 核心提取逻辑 ----------
class LogisticsExtractor:
    def __init__(self):
        self.model, self.processor = init_models()
        self.field_definition = order_fields


    def validate_and_convert(self, raw_data: dict) -> dict:
        """后处理验证和类型转换"""
        processed = {}
        for field in self.field_definition:
            value = raw_data.get(field, "")
            
            # 执行类型转换
            if "datetime" in self.field_definition[field]:
                try:
                    processed[field] = datetime.strptime(value, "%Y-%m-%d").isoformat()
                except:
                    processed[field] = ""
            elif "float" in self.field_definition[field]:
                processed[field] = float(value) if value else 0.0
            elif "integer" in self.field_definition[field]:
                processed[field] = int(value) if value else 0
            else:
                processed[field] = value.strip()
                
        return processed

    def extract_from_image(self, image_path: str) -> ExtractionResult:
        """执行完整提取流程"""
        try:
            # 输入验证
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            # 准备输入
            image = Image.open(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_prompt()}
                ]
            }]

            # 模型推理
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            
            # 清理之前的显存
            torch.cuda.empty_cache()
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # 增加token长度
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

            # 解码和后处理
            response = self.processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            # 提取并验证JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("未检测到有效JSON输出")

            raw_data = json.loads(json_match.group())
            validated_data = self.validate_and_convert(raw_data)

            return ExtractionResult(
                content_type="image",
                original_data=image_path,
                extracted_fields=validated_data,
                confidence=1.0  # 模型暂不返回置信度
            )

        except Exception as e:
            return ExtractionResult(
                content_type="image",
                original_data=image_path,
                extracted_fields={"error": str(e)},
                confidence=0.0
            )

# ---------- 主程序 ----------
if __name__ == "__main__":
    extractor = LogisticsExtractor()
    
    # 处理vision目录下的图片
    vision_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'vision'
    )
    
    results = []
    for filename in os.listdir(vision_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(vision_dir, filename)
            result = extractor.extract_from_image(image_path)
            results.append(result.model_dump()) 
    
    # 输出结果
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
 
