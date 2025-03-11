import torch
import os
import re
import json
import yaml
from datetime import datetime
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from ..schemas import order_fields

# ---------- 项目结构配置 ----------
class ExtractionResult(BaseModel):
    content_type: str
    original_data: str
    extracted_fields: dict
    confidence: float

# ---------- 配置加载 ----------
def load_config():
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "settings.yaml"))
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
SPEECH_CONFIG = config.get("text_model_config", {})

# 配置参数
CUDA_DEVICES = SPEECH_CONFIG.get("cuda_devices", "0")
MODEL_PATH = SPEECH_CONFIG.get("model_path", "Qwen/Qwen2.5-7B-Instruct")

# 设备配置
device = f"cuda:{CUDA_DEVICES}" if torch.cuda.is_available() else "cpu"

# ---------- 模型初始化 ----------
def init_models():
    """初始化文本处理模型"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        selected_gpu = min(int(CUDA_DEVICES), num_gpus-1) if CUDA_DEVICES.isdigit() else 0
        model_device = f"cuda:{selected_gpu}"
    else:
        model_device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if "cuda" in model_device else torch.float32,
        device_map={"": model_device},
        trust_remote_code=True
    )

    return model, tokenizer

# ---------- 核心处理类 ----------
class TextProcessor:
    def __init__(self):
        self.model, self.tokenizer = init_models()
        self.field_definitions = order_fields

    def _build_prompt(self, text: str) -> str:
        """构建结构化提取提示语"""
        fields_desc = "\n".join(
            [f"- {k}: {v}" for k, v in self.field_definitions.items()])
        
        return f"""请从以下文本内容中提取结构化信息：
        
【文本内容】
{text}

【提取字段】
{fields_desc}

【输出要求】
1. 返回纯净JSON，无额外字符
2. 缺失字段保留为空白
3. 严格遵循字段格式
4. 金额单位：人民币元
"""

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
                elif "list" in field:
                    processed[field] = [item.strip() for item in value.split(",")] if value else []
                else:
                    processed[field] = str(value).strip()
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
            prompt = self._build_prompt(text)
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
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # 生成输出
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # 解析结果
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("未检测到有效JSON")
                
            raw_data = json.loads(json_match.group())
            processed_data = self._postprocess_data(raw_data)

            result["data"] = {
                "original_text": text,
                "structured": processed_data
            }

        except json.JSONDecodeError as e:
            result.update({
                "status": "error",
                "error": f"JSON解析失败: {str(e)}"
            })
        except Exception as e:
            result.update({
                "status": "error",
                "error": f"处理失败: {str(e)}"
            })

        return result

# ---------- 主程序 ----------
if __name__ == "__main__":
    processor = TextProcessor()
    
    # 示例文本
    test_text = "我需要预定一辆轿运车，从上海到北京，5月20日发货，运费预算5000元左右，需要运输3辆SUV。"
    
    print(f"\n处理文本: {test_text}")
    start_time = datetime.now()
    result = processor.process(test_text)
    process_time = (datetime.now() - start_time).total_seconds()
    result["processing_time"] = f"{process_time:.2f}s"
    
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
