import torch
import os
import re
import json
import yaml
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
from ..schemas import order_fields
from ..utils.config_utils import load_config
from ..schemas import order_fields, ExtractionResult
from ..utils.prompt_utils import build_prompt
from vllm import SamplingParams
from ..utils.processor_utils import init_text_model

config = load_config()
SPEECH_CONFIG = config.get("text_model_config", {})

# 配置参数
CUDA_DEVICES = SPEECH_CONFIG.get("cuda_devices", "0")
MODEL_PATH = SPEECH_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")

# 设备配置
device = f"cuda:{CUDA_DEVICES}" if torch.cuda.is_available() else "cpu"

# ---------- 核心处理类 ----------
class TextProcessor:
    def __init__(self):
        self.model, self.tokenizer = init_text_model()
        self.field_definitions = order_fields
        # 获取当前设备
        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        # 初始化采样参数
        self.sampling_params = SamplingParams(
            temperature=0.3,  # 降低温度以提高输出的确定性
            top_p=0.9,      # 降低top_p以减少输出的随机性
            max_tokens=4096,  # 增加最大token数以确保完整输出
            presence_penalty=0.0,  # 移除存在惩罚
            frequency_penalty=0.0  # 移除频率惩罚
        )

    def _postprocess_data(self, raw_data: dict) -> dict:
        """数据后处理（与原始实现保持一致）"""
        processed = {}
        for field in self.field_definitions:
            value = raw_data.get(field, "")
            
            try:
                if "date" in field:
                    processed[field] = datetime.strptime(value, "%Y-%m-%d").date().isoformat() if value else ""
                elif "amount" in field:
                    processed[field] = float(value) if value else 0.0
                elif "integer" in self.field_definitions[field]:
                    processed[field] = int(value) if value else 0
                elif "array" in self.field_definitions[field]:
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
            except Exception:
                processed[field] = value  # 保留原始值用于调试

        return processed

    def process(self, text: str) -> Dict[str, Any]:
        """完整处理流程"""
        result = {
            "status": "success",
            "data": {},
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # 结构化提取
            prompt = build_prompt(text)
            messages = [
                {"role": "system", "content": "你是一个专业的信息提取助手，请严格按用户要求输出JSON格式。"},
                {"role": "user", "content": prompt}
            ]
            
            # 生成输入
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 生成输出
            outputs = self.model.generate(
                text,
                self.sampling_params,

            )
            response = ""
            for output in outputs:
                response += output.outputs[0].text
            
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                # 尝试查找不完整的JSON
                incomplete_json = re.search(r'\{[\s\S]*', response, re.DOTALL)
                if incomplete_json:
                    json_str = incomplete_json.group()
                    # 检查是否缺少右花括号
                    if not json_str.rstrip().endswith('}'):
                        # 尝试修复JSON
                        json_str = json_str.rstrip() + '}'
                raise ValueError("未检测到有效JSON")
                
            raw_data = json.loads(json_match.group())
            processed_data = self._postprocess_data(raw_data)

            result["data"] = {
                "original_text": text,
                "structured": processed_data
            }
            return ExtractionResult(
                content_type="text",
                extracted_fields=processed_data,
                confidence=1.0  # 模型暂不返回置信度
            )

        except json.JSONDecodeError as e:
            print("\n=== JSON解析错误 ===")
            print("错误信息:", str(e))
            print("原始Response:", response)
            return ExtractionResult(
                content_type="text",
                extracted_fields={"error": str(e)},
                confidence=0.0
            )
        except Exception as e:
            print("\n=== 其他错误 ===")
            print("错误信息:", str(e))
            print("原始Response:", response)
            return ExtractionResult(
                content_type="text",
                extracted_fields={"error": str(e)},
                confidence=0.0
            )

# ---------- 主程序 ----------
if __name__ == "__main__":
    processor = TextProcessor()
    
    # 示例文本
    test_text = "我需要预定一辆轿运车，从上海到北京，5月20日发货，运费预算5000元左右，需要运输3辆SUV。"
    
    print(f"\n处理文本: {test_text}")
    start_time = datetime.now()
    result = processor.process(test_text)
    process_time = (datetime.now() - start_time).total_seconds()
    
    # 将 ExtractionResult 转换为字典
    result_dict = result.dict()
    result_dict["processing_time"] = f"{process_time:.2f}s"
    
    print(json.dumps(result_dict, indent=2, ensure_ascii=False, default=str))
