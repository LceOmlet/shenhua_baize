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
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import imageio_ffmpeg as ffmpeg


# ---------- 根据项目结构调整的模块导入 ----------
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pydantic import BaseModel
from schemas import order_fields
class ExtractionResult(BaseModel):
    content_type: str
    original_data: str
    extracted_fields: dict
    confidence: float  # 从项目根目录导入


# ---------- 加载配置文件 ----------
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "settings.yaml"))
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# 读取配置
config = load_config()
SPEECH_MODEL_CONFIG = config.get("speech_model_config", {})


# 获取 CUDA 设备配置
CUDA_DEVICES = SPEECH_MODEL_CONFIG.get("cuda_devices", "0")  # 默认使用 GPU 0
MODEL_PATH = SPEECH_MODEL_CONFIG.get("model_path", "Qwen/Qwen2-Audio-7B-Instruct")
SAMPLE_RATE = SPEECH_MODEL_CONFIG.get("sample_rate", 16000)

# 获取设备
device = f"cuda:{CUDA_DEVICES}" if torch.cuda.is_available() else "cpu"




# ---------- 初始化模型 ----------
def init_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)


    # 获取 `CUDA_DEVICES` 并确保其合法
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        device = "cpu"
    else:
        selected_gpu = int(CUDA_DEVICES) if CUDA_DEVICES.isdigit() else 0
        if selected_gpu >= num_gpus:
            selected_gpu = 0  # 超界时回退到 GPU 0
        device = f"cuda:{selected_gpu}"

    # **手动指定 `device_map`**
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map={"": device}  # 手动分配设备
    )

    model.tie_weights()
    return model, processor


# ---------- AMR 转 WAV ----------
def convert_amr_to_wav(amr_path: str) -> str:
    wav_path = amr_path.replace(".amr", ".wav")
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    # 获取原始时间戳
    original_mtime = os.path.getmtime(amr_path)  # 获取修改时间

    try:
        subprocess.run(
            [ffmpeg_path, "-i", amr_path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 还原时间戳
        os.utime(wav_path, (original_mtime, original_mtime))  # 设置 WAV 文件的修改时间

        # 删除原 AMR 文件
        os.remove(amr_path)

        # 用 WAV 替换 AMR（重命名 WAV）
        new_amr_path = amr_path.replace(".amr", ".wav")  # 保证名称一致
        os.rename(wav_path, new_amr_path)

        return new_amr_path
    except subprocess.CalledProcessError as e:
        print(f"AMR 转换失败: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"文件处理错误: {str(e)}")
        return None



# ---------- 音频处理类 ----------
class AudioProcessor:
    def __init__(self):
        self.model, self.processor = init_model()  # **在这里不会报错**
        self.field_definition = order_fields

    def build_prompt(self):
        """动态构建包含所有字段要求的提示语"""
        field_descriptions = []
        for field, desc in self.field_definition.items():
            field_descriptions.append(f"- {field}（{desc}）")

        # 使用显式换行符拼接
        prompt_lines = [
            "<|AUDIO|>",
            "请严格分析音频，按以下要求提取字段：",
            *[f"- {field}（{desc}）" for field, desc in self.field_definition.items()],
            "",
            "规则：",
            "1. 缺失字段返回空字符串",
            "2. 严格遵循指定格式",
            "3. 输出纯净JSON，不要额外字符",
            "4. 可以不用全部填满"
        ]
        return "\n".join(prompt_lines)

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

    def load_audio(self, audio_path: str) -> Dict[str, Any]:
        """加载音频文件"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        try:
            waveform, sr = torchaudio.load(audio_path)  # `waveform` 形状: (通道数, 采样点数)

            # **确保转换为单通道（如果是立体声，取均值）**
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # 取平均，转换成单声道

            # **如果采样率不同，则进行重新采样**
            if sr != SAMPLE_RATE:
                resampler = transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            # **转换为 numpy 格式**
            audio_data = waveform.numpy().squeeze()  # `squeeze()` 只移除 batch 维度，不影响时间维度

            # **计算音频时长**
            duration = audio_data.shape[-1] / SAMPLE_RATE


        except Exception as e:
            raise ValueError(f"无法读取音频文件 {audio_path}: {e}")

        return {
            "audio": np.asarray(audio_data, dtype=np.float32),
            "sampling_rate": SAMPLE_RATE,
            "duration": duration  # 记录音频时长
        }

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """执行完整提取流程"""
        try:
            audio_info = self.load_audio(audio_path)

            if audio_info["audio"] is None or len(audio_info["audio"]) == 0:
                raise ValueError("音频数据为空，无法处理")

            with torch.no_grad():
                # **构造对话历史**
                chat_history = [
                    {"role": "user", "content": [
                        {"type": "text", "text": self.build_prompt()},
                        {"type": "audio", "audio_path": audio_path}
                    ]}
                ]

                # **格式化文本**
                formatted_text = self.processor.apply_chat_template(
                    chat_history, add_generation_prompt=True, tokenize=False
                )

                # **生成 `inputs`**
                inputs = self.processor(
                    text=formatted_text,
                    audios=[audio_info["audio"]],
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=SAMPLE_RATE  # 确保 `sampling_rate` 正确传递
                )

                # 移动 `inputs` 到 `cuda`
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # 确保模型在 `cuda`
                self.model.to(device)

                # **执行 `generate()`**
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

                # **解码新生成的文本**
                transcription = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # 提取并验证JSON
                json_match = re.search(r'\{.*\}', transcription, re.DOTALL)
                if not json_match:
                    raise ValueError("未检测到有效JSON输出")

                raw_data = json.loads(json_match.group())
                validated_data = self.validate_and_convert(raw_data)

            return {
                "content_type": "audio",
                "original_data": audio_path,
                "transcription": validated_data,
                "confidence": 1.0,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "content_type": "audio",
                "original_data": audio_path,
                "error": str(e),
                "confidence": 0.0
            }


# ---------- 运行主程序 ----------
if __name__ == "__main__":
    processor = AudioProcessor()
    audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "audio"))
    results = []

    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.amr')):
            audio_path = os.path.join(audio_dir, filename)
            if filename.endswith('.amr'):
                audio_path = convert_amr_to_wav(audio_path)

            if audio_path:
                result = processor.transcribe_audio(audio_path)
                results.append(result)
    print(json.dumps(results, indent=2, ensure_ascii=False))
