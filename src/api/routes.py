from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import sys
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import torch
import gc
import time

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# 添加src目录到Python路径
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from src.processing.audio_processor import AudioProcessor
from src.processing.image_processor import LogisticsExtractor
from src.utils.file_utils import unified_process

app = FastAPI(
    title="白泽",
    description="图像和音频分析工具"
)

# 全局变量存储处理器实例
audio_processor = None
image_processor = None

# 创建线程池
executor = ThreadPoolExecutor(max_workers=1)  # 减少并发数

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def load_audio_model():
    """加载音频模型"""
    global audio_processor
    if audio_processor is None:
        print("正在加载音频模型...")
        clear_gpu_memory()  # 加载前清理内存
        audio_processor = AudioProcessor()
    return audio_processor

def load_image_model():
    """加载图像模型"""
    global image_processor
    if image_processor is None:
        print("正在加载图像模型...")
        clear_gpu_memory()  # 加载前清理内存
        image_processor = LogisticsExtractor()
    return image_processor

def release_audio_model():
    """释放音频模型资源"""
    global audio_processor
    if audio_processor is not None:
        print("正在释放音频模型资源...")
        # 清理模型资源
        if hasattr(audio_processor, 'model'):
            del audio_processor.model
        if hasattr(audio_processor, 'whisper_model'):
            del audio_processor.whisper_model
        audio_processor = None
        clear_gpu_memory()

def release_image_model():
    """释放图像模型资源"""
    global image_processor
    if image_processor is not None:
        print("正在释放图像模型资源...")
        # 清理模型资源
        if hasattr(image_processor, 'model'):
            del image_processor.model
        image_processor = None
        clear_gpu_memory()

async def process_with_retry(processor, process_func, file_path, max_retries=3):
    """带重试机制的处理函数"""
    for attempt in range(max_retries):
        try:
            # 每次尝试前清理内存
            clear_gpu_memory()
            
            # 使用线程池处理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                process_func,
                file_path
            )
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < max_retries - 1:
                print(f"内存不足，尝试 {attempt + 1}/{max_retries}")
                clear_gpu_memory()
                time.sleep(1)  # 等待一秒后重试
                continue
            raise
        except Exception as e:
            raise

@app.get("/")
async def root():
    return {
        "系统名称": "白泽",
        "版本": "1.0.0",
        "功能": {
            "图像分析": "/image",
            "音频分析": "/audio",
            "自动识别处理": "/auto"
        },
        "系统状态": {
            "运行状态": "正常",
            "模型状态": {
                "图像模型": "已加载" if image_processor else "未加载",
                "音频模型": "已加载" if audio_processor else "未加载"
            }
        }
    }

@app.post("/image", summary="图像分析")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 加载图像模型
        processor = load_image_model()
        
        # 保存上传的文件
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用重试机制处理图像
        result = await process_with_retry(
            processor,
            processor.extract_from_image,
            file_path
        )
        
        # 清理临时文件
        os.remove(file_path)
        
        # 释放模型资源
        release_image_model()
        
        return {"分析结果": result.model_dump()}
    except Exception as e:
        # 确保发生错误时也释放资源
        release_image_model()
        return {"错误": str(e)}

@app.post("/audio", summary="音频分析")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 加载音频模型
        processor = load_audio_model()
        
        # 保存上传的文件
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用重试机制处理音频
        result = await process_with_retry(
            processor,
            processor.process,
            file_path
        )
        
        # 清理临时文件
        os.remove(file_path)
        
        # 释放模型资源
        release_audio_model()
        
        return {"分析结果": result.model_dump()}
    except Exception as e:
        # 确保发生错误时也释放资源
        release_audio_model()
        return {"错误": str(e)}

@app.post("/auto", summary="自动识别与处理")
async def auto_process(file: UploadFile = File(...)):
    """
    自动识别上传文件类型并进行相应处理
    支持音频、图像和文本文件
    """
    try:
        # 保存上传的文件
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 使用统一处理函数进行自动识别和处理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            unified_process,
            file_path
        )
        
        # 清理临时文件
        os.remove(file_path)
        
        # 根据处理结果决定是否释放模型
        # 注意：统一处理函数内部会加载需要的模型
        if audio_processor is not None:
            release_audio_model()
        if image_processor is not None:
            release_image_model()
        
        # 检查处理结果是否有错误
        if isinstance(result, dict) and result.get("success") is False:
            return {"状态": "失败", "错误详情": result.get("error"), "结果": None}
        
        # 处理成功的情况
        if hasattr(result, "model_dump"):
            return {"状态": "成功", "结果": result.model_dump()}
        return {"状态": "成功", "结果": result}
        
    except Exception as e:
        # 确保发生错误时也释放资源
        if audio_processor is not None:
            release_audio_model()
        if image_processor is not None:
            release_image_model()
            
        # 移除临时文件（如果存在）
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        return {"状态": "失败", "错误": str(e), "结果": None}

@app.get("/status", summary="系统状态")
async def get_status():
    # 获取系统信息
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 获取GPU信息
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "设备数量": torch.cuda.device_count(),
            "当前设备": torch.cuda.current_device(),
            "设备名称": torch.cuda.get_device_name(0),
            "显存使用": f"{torch.cuda.memory_allocated(0)/1024**2:.2f}MB",
            "显存总量": f"{torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB",
            "显存缓存": f"{torch.cuda.memory_reserved(0)/1024**2:.2f}MB"
        }
    
    return {
        "系统信息": {
            "CPU使用率": f"{cpu_percent}%",
            "内存使用": f"{memory.percent}%",
            "磁盘使用": f"{disk.percent}%"
        },
        "模型状态": {
            "图像模型": "已加载" if image_processor else "未加载",
            "音频模型": "已加载" if audio_processor else "未加载"
        },
        "GPU信息": gpu_info if gpu_info else "未检测到GPU",
        "运行状态": "正常"
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8070"))
    print(f"服务器启动中... 访问地址: http://{host}:{port}")
    print("注意：模型将在首次使用时加载，使用后自动释放")
    uvicorn.run(app, host=host, port=port) 