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
from typing import List
from PIL import Image
import numpy as np
from pydub import AudioSegment
import io
from pdf2image import convert_from_path

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
            
            # 直接调用异步函数
            result = await process_func(file_path)
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

def merge_images(image_paths: List[str], output_path: str):
    """合并多个图片文件"""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    
    # 计算合并后的图片尺寸
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # 如果合并后的图片太大，进行压缩
    if total_width > 2048 or max_height > 2048:
        # 计算压缩比例
        scale = min(2048 / total_width, 2048 / max_height)
        new_width = int(total_width * scale)
        new_height = int(max_height * scale)
        
        # 创建新图片
        merged_image = Image.new('RGB', (new_width, new_height))
        
        # 拼接并压缩图片
        x_offset = 0
        for img in images:
            # 压缩单个图片
            img_width = int(img.width * scale)
            img_height = int(img.height * scale)
            resized_img = img.resize((img_width, img_height), Image.LANCZOS)
            merged_image.paste(resized_img, (x_offset, 0))
            x_offset += img_width
    else:
        # 创建新图片
        merged_image = Image.new('RGB', (total_width, max_height))
        
        # 拼接图片
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
    
    # 保存合并后的图片
    merged_image.save(output_path, 'JPEG', quality=85)
    return output_path

def merge_audio(audio_paths: List[str], output_path: str):
    """合并多个音频文件"""
    combined = AudioSegment.empty()
    
    for path in audio_paths:
        audio = AudioSegment.from_file(path)
        combined += audio
    
    # 保存合并后的音频
    combined.export(output_path, format=output_path.split('.')[-1])
    return output_path

def convert_pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """将PDF文件转换为图片"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换PDF为图片
    images = convert_from_path(pdf_path)
    image_paths = []
    
    # 保存每一页为图片
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
        # 压缩图片
        compressed_image = compress_image(image)
        compressed_image.save(image_path, 'JPEG', quality=85)
        image_paths.append(image_path)
    
    return image_paths

def compress_image(image, max_size: int = 1024, quality: int = 85):
    """
    压缩图片，减小尺寸和质量
    
    Args:
        image: PIL Image 对象
        max_size: 最大尺寸（宽度或高度）
        quality: JPEG 质量（1-100）
        
    Returns:
        压缩后的图片
    """
    # 计算新的尺寸
    width, height = image.size
    if width > height:
        new_width = min(width, max_size)
        new_height = int(height * (new_width / width))
    else:
        new_height = min(height, max_size)
        new_width = int(width * (new_height / height))
    
    # 调整图像大小
    compressed = image.resize((new_width, new_height), Image.LANCZOS)
    
    return compressed

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

@app.post("/auto", summary="自动识别与处理")
async def auto_process(files: List[UploadFile] = File(...)):
    """
    自动识别上传文件类型并进行相应处理
    支持多个音频、图像、文本和PDF文件合并处理
    """
    temp_files = []
    try:
        # 创建临时目录
        os.makedirs("temp", exist_ok=True)
        
        # 保存所有上传的文件
        for file in files:
            file_path = f"temp/{file.filename}"
            temp_files.append(file_path)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        # 按文件类型分组
        text_files = []
        image_files = []
        audio_files = []
        pdf_files = []
        
        for file_path in temp_files:
            if file_path.lower().endswith(('.txt', '.json', '.csv')):
                text_files.append(file_path)
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file_path)
            elif file_path.lower().endswith(('.wav', '.mp3', '.amr')):
                audio_files.append(file_path)
            elif file_path.lower().endswith('.pdf'):
                pdf_files.append(file_path)
        
        # 处理PDF文件，转换为图片
        for pdf_path in pdf_files:
            pdf_images = convert_pdf_to_images(pdf_path, "temp/pdf_images")
            image_files.extend(pdf_images)
        
        # 根据文件类型选择处理方式
        if len(text_files) > 0:
            # 处理文本文件
            combined_file = text_files[0]  # 使用第一个文本文件作为基础
            if len(text_files) > 1:
                # 如果有多个文本文件，合并内容
                with open(combined_file, "a", encoding="utf-8") as f:
                    for file_path in text_files[1:]:
                        with open(file_path, "r", encoding="utf-8") as src:
                            f.write("\n" + src.read())
        elif len(image_files) > 0:
            # 处理图片文件
            if len(image_files) > 1:
                # 如果有多个图片文件，合并它们
                combined_file = "temp/merged_image.jpg"
                combined_file = merge_images(image_files, combined_file)
            else:
                combined_file = image_files[0]
        elif len(audio_files) > 0:
            # 处理音频文件
            if len(audio_files) > 1:
                # 如果有多个音频文件，合并它们
                combined_file = "temp/merged_audio.wav"
                combined_file = merge_audio(audio_files, combined_file)
            else:
                combined_file = audio_files[0]
        else:
            raise ValueError("不支持的文件类型")
        
        # 使用统一处理函数处理合并后的文件
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            unified_process,
            combined_file
        )
        
        # 清理所有临时文件
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if combined_file.startswith("temp/merged_"):
            os.remove(combined_file)
        # 清理PDF转换的图片
        if os.path.exists("temp/pdf_images"):
            for file in os.listdir("temp/pdf_images"):
                os.remove(os.path.join("temp/pdf_images", file))
            os.rmdir("temp/pdf_images")
        
        # 根据处理结果决定是否释放模型
        if audio_processor is not None:
            release_audio_model()
        if image_processor is not None:
            release_image_model()
        
        # 检查处理结果是否有错误
        if isinstance(result, dict) and result.get("success") is False:
            return {
                "状态": "失败",
                "错误详情": result.get("error"),
                "处理文件数": len(files),
                "结果": None
            }
        
        # 处理成功的情况
        if hasattr(result, "model_dump"):
            return {
                "状态": "成功",
                "处理文件数": len(files),
                "结果": result.model_dump()
            }
        return {
            "状态": "成功",
            "处理文件数": len(files),
            "结果": result
        }
        
    except Exception as e:
        # 确保发生错误时也释放资源
        if audio_processor is not None:
            release_audio_model()
        if image_processor is not None:
            release_image_model()
            
        # 移除所有临时文件
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if 'combined_file' in locals() and combined_file.startswith("temp/merged_"):
            os.remove(combined_file)
        # 清理PDF转换的图片
        if os.path.exists("temp/pdf_images"):
            for file in os.listdir("temp/pdf_images"):
                os.remove(os.path.join("temp/pdf_images", file))
            os.rmdir("temp/pdf_images")
            
        return {
            "状态": "失败",
            "错误": str(e),
            "处理文件数": len(files),
            "结果": None
        }

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