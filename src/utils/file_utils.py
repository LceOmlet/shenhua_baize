import os
import shutil
import yaml
import time
from werkzeug.utils import secure_filename

# 加载配置
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "settings.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

STORAGE_CONFIG = config.get("storage_config", {})
TEMP_DIR = STORAGE_CONFIG.get("temp_dir", "temp")
MAX_FILE_SIZE = STORAGE_CONFIG.get("max_file_size", 10) * 1024 * 1024  # 10MB
EXPIRE_TIME = STORAGE_CONFIG.get("expire_time", 3600)  # 1小时

# 确保 temp 目录存在
os.makedirs(TEMP_DIR, exist_ok=True)

def save_file(file):
    """存储上传文件到临时目录"""
    filename = secure_filename(file.name)
    file_path = os.path.join(TEMP_DIR, filename)

    # 检查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise ValueError("文件大小超出限制")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file, f)

    return file_path

def clean_old_files():
    """定期清理过期的文件"""
    now = time.time()
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > EXPIRE_TIME:
            os.remove(file_path)
