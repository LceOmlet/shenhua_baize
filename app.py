import streamlit as st
import os
from src.utils.file_utils import save_file, clean_old_files
from src.schemas import ExtractionResult
from src.processing.text_processor import TextProcessor
from src.processing.audio_processor import AudioProcessor
from src.processing.image_processor import LogisticsExtractor

# 初始化处理器（使用持久化的模型）
text_processor = TextProcessor()
audio_processor = AudioProcessor()
image_processor = LogisticsExtractor()

# 界面设置
st.title("多模态信息提取器")

# 文件上传
uploaded_files = st.file_uploader(
    "拖拽文件到这里 (图片/音频/文本)",
    type=["jpg", "png", "wav", "mp3", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    saved_files = []
    for file in uploaded_files:
        try:
            file_path = save_file(file)
            saved_files.append(file_path)
            st.success(f"已上传: {file.name}")

            # 处理不同类型的文件
            if file.type.startswith("image"):  # 处理图像
                result = image_processor.extract_from_image(file_path)
            elif file.type.startswith("audio"):  # 处理音频
                result = audio_processor.process(file_path)
            elif file.type.startswith("text"):  # 处理文本
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                result = text_processor.process(text_content)
            else:
                result = {"error": "不支持的文件类型"}

            st.json(result)  # 在前端显示处理结果

        except ValueError as e:
            st.error(str(e))

# 显示已上传的文件
st.subheader("已上传文件")
temp_dir = "temp"
if os.path.exists(temp_dir):
    files = os.listdir(temp_dir)
    for file in files:
        file_path = os.path.join(temp_dir, file)
        st.write(f"📄 {file}")
        if st.button(f"🗑 删除 {file}", key=file):
            os.remove(file_path)
            st.rerun()

# 清理过期文件
clean_old_files()
