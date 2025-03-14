import torch
import torchaudio
import torchaudio.transforms as transforms
import os
import re
import json
import yaml
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import imageio_ffmpeg as ffmpeg
from pydantic import BaseModel
from ..schemas import order_fields, ExtractionResult
from ..utils.model_utils import SPEECH_MODEL
from ..utils.config_utils import load_config


config = load_config()
SPEECH_CONFIG = config.get("speech_model_config", {})
WHISPER_CONFIG = config.get("whisper_config", {})

# 配置参数
SAMPLE_RATE = SPEECH_CONFIG.get("sample_rate", 16000)
WHISPER_PROMPT = WHISPER_CONFIG.get("prompt", "")


# ---------- 模型初始化 ----------
def init_models():
    return SPEECH_MODEL

# ---------- 音频格式转换 ----------
def convert_amr_to_wav(amr_path: str) -> str:
    """AMR转WAV格式"""
    wav_path = amr_path.replace(".amr", ".wav")
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    try:
        subprocess.run(
            [ffmpeg_path, "-i", amr_path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.remove(amr_path)
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"文件错误: {str(e)}")
        return None

# ---------- 核心处理类 ----------
class AudioProcessor:
    def __init__(self):
        self.model, self.tokenizer, self.whisper_model = init_models()
        self.field_definitions = order_fields

    def _build_prompt(self, transcription: str) -> str:
        """构建结构化提取提示语"""
        fields_desc = "\n".join(
            [f"- {k}: {v}" for k, v in self.field_definitions.items()])
        
        return f"""请从以下语音转录内容中提取结构化信息：
        
【转录内容】
{transcription}

【提取字段】
{fields_desc}

【输出要求】
1. 返回纯净JSON，无额外字符
2. 缺失字段保留为空白
3. 严格遵循字段格式
4. 金额单位：人民币元
"""

    def _postprocess_data(self, raw_data: dict) -> dict:
        """数据后处理"""
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

    def _load_audio(self, path: str) -> np.ndarray:
        """加载并预处理音频"""
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform.numpy().squeeze()

    def process(self, audio_path: str) -> Dict[str, Any]:
        """完整处理流程"""
        result = {
            "status": "success",
            "data": {},
            "error": None,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0
        }

        try:
            # 1. 格式转换
            if audio_path.endswith(".amr"):
                converted_path = convert_amr_to_wav(audio_path)
                if not converted_path:
                    raise ValueError("AMR转换失败")
                audio_path = converted_path

            # 2. 语音识别
            whisper_result = self.whisper_model.transcribe(
                audio_path,
                initial_prompt=WHISPER_PROMPT,
                language="zh"
            )
            transcription = whisper_result["text"]
            result["confidence"] = whisper_result.get("avg_logprob", 0.0)

            # 3. 结构化提取
            prompt = self._build_prompt(transcription)
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
                "transcription": transcription,
                "structured": processed_data
            }
            
            return ExtractionResult(
                content_type="image",
                original_data=audio_path,
                extracted_fields=processed_data,
                confidence=1.0  # 模型暂不返回置信度
            )
            

        except json.JSONDecodeError as e:
            return ExtractionResult(
                content_type="image",
                original_data=audio_path,
                extracted_fields={"error": str(e)},
                confidence=0.0
            )
        except Exception as e:
            return ExtractionResult(
                content_type="image",
                original_data=audio_path,
                extracted_fields={"error": str(e)},
                confidence=0.0
            )

# ---------- 主程序 ----------
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # 示例音频路径
    test_dir = os.path.join(os.path.dirname(__file__), "audio")
    results = []
    
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.wav', '.mp3', '.amr')):
            file_path = os.path.join(test_dir, file)
            print(f"\n处理文件: {file}")
            
            start_time = datetime.now()
            result = processor.process(file_path)
            process_time = (datetime.now() - start_time).total_seconds()
            
            result["processing_time"] = f"{process_time:.2f}s"
            results.append(result)
            
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # 保存完整结果
    output_path = os.path.join(test_dir, "processing_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
