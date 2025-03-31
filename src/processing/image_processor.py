# logistics_vision.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import PIL
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
from ..utils.prompt_utils import build_prompt
from vllm import SamplingParams
from vllm.inputs import TextPrompt


# ---------- 根据项目结构调整的模块导入 ----------
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pydantic import BaseModel
from ..utils.processor_utils import init_vision_model

# ---------- 核心提取逻辑 ----------
class LogisticsExtractor:
    def __init__(self):
        self.model, self.processor = init_vision_model()
        self.field_definition = order_fields
        # 初始化采样参数
        self.sampling_params = SamplingParams(
            temperature=0.3,  # 降低温度以提高输出的确定性
            top_p=0.9,      # 降低top_p以减少输出的随机性
            max_tokens=6144,  # 增加最大token数以确保完整输出
            presence_penalty=0.0,  # 移除存在惩罚
            frequency_penalty=0.0  # 移除频率惩罚
        )

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
            elif "array" in self.field_definition[field]:
                # 处理数组类型
                if isinstance(value, list):
                    processed[field] = value
                elif isinstance(value, str):
                    processed[field] = [item.strip() for item in value.split(",")] if value else []
                else:
                    processed[field] = []
            else:
                # 处理字符串类型
                if isinstance(value, str):
                    processed[field] = value.strip()
                else:
                    processed[field] = str(value)
                
        return processed

    def extract_from_image(self, image_path: str) -> ExtractionResult:
        """执行完整提取流程"""
        response = ""  # 初始化response变量
        try:
            # 输入验证
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            # 准备输入
            image = PIL.Image.open(image_path)
            messages = [
                {
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
            
            # 清理之前的显存
            torch.cuda.empty_cache()
            multi_modal_data = {
                "image": [image]
            }

            prompt = TextPrompt(
                prompt=text,
                multi_modal_data=multi_modal_data
            )
            
            # 生成输出
            print("\n=== 开始生成输出 ===")

            response = ""
            outputs = self.model.generate(
                prompt,
                self.sampling_params,
            )

            for output in outputs:
                response += output.outputs[0].text

            # 检查响应是否为空
            if not response:
                print("\n=== 警告: 模型输出为空 ===")
                raise ValueError("模型输出为空")

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
            print("\n=== 错误 ===")
            print("错误信息:", str(e))
            print("原始Response:", response)
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